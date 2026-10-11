"""Tests for the persistent event log controller."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from music_assistant_models.auth import User, UserRole
from music_assistant_models.enums import EventType, MediaType
from music_assistant_models.event import MassEvent
from music_assistant_models.favorite_update import FavoriteUpdate
from music_assistant_models.media_items import ProviderMapping, Track

from music_assistant.controllers.eventlog import EventLogController, ResumableSubscription
from music_assistant.controllers.eventlog.controller import EventRecord, EventWindowTrimmedError

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant


def _event(
    event: EventType = EventType.PLAYER_UPDATED,
    object_id: str | None = "p1",
    data: Any = None,
) -> MassEvent:
    return MassEvent(event=event, object_id=object_id, data=data)


class _FakeListenerClient:
    """Capture records a subscription hands out while live."""

    def __init__(self) -> None:
        self.received: list[int] = []

    def _deliver_event_record(
        self, subscription: ResumableSubscription, record: EventRecord
    ) -> None:
        self.received.append(record.seq)


async def test_admit_assigns_monotonic_sequence_and_fans_out(
    event_log: EventLogController,
) -> None:
    """Admitted events get strictly increasing seqs and reach live listeners."""
    client = _FakeListenerClient()
    subscription = ResumableSubscription(client=client, buffering=False)
    head = event_log.register_listener(subscription)
    assert head == 0

    event_log.admit(_event(object_id="p1"))
    event_log.admit(_event(object_id="p2"))
    await asyncio.sleep(0)  # run the call_soon fan-outs

    assert client.received == [1, 2]
    assert event_log.head_seq == 2


async def test_admit_is_noop_before_ready(mass_minimal: MusicAssistant) -> None:
    """An event log that was never set up ignores admission."""
    controller = EventLogController(mass_minimal)
    assert controller.admit(_event()) is None
    assert controller.head_seq == 0


async def test_payload_freeze_and_rehydrate_types(event_log: EventLogController) -> None:
    """Typed payloads round-trip through the window with their identity preserved."""
    favorite = FavoriteUpdate(
        uri="lib/t", media_type=MediaType.TRACK, item_id="t", favorite=True, user_id="u1"
    )
    track = Track(
        item_id="t1",
        provider="library",
        name="Song",
        duration=100,
        provider_mappings={
            ProviderMapping(
                item_id="t1", provider_domain="filesystem_local", provider_instance="fs"
            )
        },
    )
    event_log.admit(_event(EventType.FAVORITE_UPDATED, "lib/t", favorite))
    event_log.admit(_event(EventType.MEDIA_ITEM_UPDATED, track.uri, track))
    # plain dict payloads are frozen as nested dicts, primitives/None as raw
    event_log.admit(_event(EventType.PLAYER_UPDATED, "p1", {"state": "playing"}))
    event_log.admit(_event(EventType.MUSIC_SYNC_COMPLETED, None, None))

    records = event_log.iter_window(0)
    assert [r.data_type for r in records] == [
        "favorite_update",
        "media_item",
        "dict",
        "raw",
    ]

    rehydrated = [event_log.rehydrate_record(record) for record in records]
    assert isinstance(rehydrated[0].data, FavoriteUpdate)
    assert rehydrated[0].data.user_id == "u1"
    assert type(rehydrated[1].data).__name__ == "Track"
    assert rehydrated[1].data.item_id == "t1"
    assert rehydrated[2].data == {"state": "playing"}
    assert rehydrated[3].data is None
    for record, event in zip(records, rehydrated, strict=True):
        # cached after first rebuild and equivalent on a second rebuild
        assert event_log.rehydrate_record(record) is event


async def test_retention_bounds_trim_the_window(
    event_log: EventLogController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capacity pruning makes old cursors invalid while recent events stay readable."""
    monkeypatch.setattr("music_assistant.controllers.eventlog.controller.MAX_WINDOW_EVENTS", 3)
    monkeypatch.setattr("music_assistant.controllers.eventlog.controller.PRUNE_EVERY_N_INSERTS", 1)
    for index in range(1, 7):
        event_log.admit(_event(object_id=f"p{index}"))

    assert event_log.oldest_seq == 4
    # cursor 3 sits exactly one before the oldest record and can still be resumed
    assert [record.seq for record in event_log.iter_window(3)] == [4, 5, 6]
    with pytest.raises(EventWindowTrimmedError):
        event_log.iter_window(2)


async def test_ack_floor_needs_every_active_connection(
    event_log: EventLogController, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One connection's ack can never reclaim events an other connection hasn't acked."""
    monkeypatch.setattr("music_assistant.controllers.eventlog.controller.MAX_WINDOW_EVENTS", 10_000)
    monkeypatch.setattr("music_assistant.controllers.eventlog.controller.PRUNE_EVERY_N_INSERTS", 1)
    client_a = _FakeListenerClient()
    client_b = _FakeListenerClient()
    sub_a = ResumableSubscription(client=client_a, buffering=False, token="tok-a")
    sub_b = ResumableSubscription(client=client_b, buffering=False, token="tok-b")
    event_log.register_listener(sub_a)
    event_log.register_listener(sub_b)

    for _ in range(3):
        event_log.admit(_event())
    await asyncio.sleep(0)

    # only connection a acknowledges everything: the global floor stays at 0
    sub_a.acked_watermark = 3
    event_log.admit(_event())
    assert event_log.oldest_seq == 1

    # once both connections are past seq 3, those events may be reclaimed
    sub_b.acked_watermark = 3
    event_log.admit(_event())
    assert event_log.oldest_seq == 4


async def test_window_and_subscription_survive_restart(
    mass_minimal: MusicAssistant,
) -> None:
    """A new controller on the same database continues the sequence and replays rows."""
    first = EventLogController(mass_minimal)
    await first.setup(None)  # type: ignore[arg-type]
    first.admit(_event(EventType.QUEUE_UPDATED, "q1"))
    first.admit(_event(EventType.QUEUE_TIME_UPDATED, "q1"))
    user = User(user_id="u1", username="u1", role=UserRole.ADMIN)
    token = await first.create_subscription(user, "scope-signature")
    await first.close()

    second = EventLogController(mass_minimal)
    await second.setup(None)  # type: ignore[arg-type]
    try:
        assert second.head_seq == 2
        records = second.iter_window(0)
        assert [record.seq for record in records] == [1, 2]
        replayed = second.rehydrate_record(records[0])
        assert replayed.event == EventType.QUEUE_UPDATED
        # the resume token survived the restart as well
        decision, _ = await second.validate_resume(
            token, 1, second.mass.server_id, user, "scope-signature", 2
        )
        assert decision.value == "resumed"
    finally:
        await second.close()


async def test_broken_payload_falls_back_to_raw(event_log: EventLogController) -> None:
    """A payload that cannot be rehydrated is delivered raw, keeping the sequence intact."""
    record = event_log.admit(_event())
    assert record is not None
    record.data_type = "model:nonexistent.module.Something"
    record.data = {"x": 1}
    record.cached_event = None
    event = event_log.rehydrate_record(record)
    assert event.data == {"x": 1}
    assert event.event == EventType.PLAYER_UPDATED
