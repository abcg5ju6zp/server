"""Tests for the resumable event stream over a websocket connection."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, cast

from music_assistant_models.api import CommandMessage
from music_assistant_models.auth import User, UserRole
from music_assistant_models.enums import EventType

from music_assistant.controllers.eventlog import compute_scope_sig
from music_assistant.controllers.eventlog.controller import EventRecord
from music_assistant.controllers.webserver.websocket_client import (
    EVENT_STREAM_ACK,
    EVENT_STREAM_SUBSCRIBE,
)

from .conftest import create_client, drain_wire, send_command, settle

if TYPE_CHECKING:
    from music_assistant.controllers.eventlog import EventLogController
    from music_assistant.mass import MusicAssistant


def _result(message: dict[str, Any]) -> dict[str, Any]:
    """Return the result payload of a SuccessResultMessage."""
    assert "result" in message, message
    return cast("dict[str, Any]", message["result"])


def _envelopes(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pull sequenced event envelopes out of a message list."""
    return [message for message in messages if "seq" in message]


async def test_fresh_subscribe_serves_snapshot_then_live_events(
    event_log_ws: dict[str, Any],
    mass_minimal: MusicAssistant,
) -> None:
    """A first subscription responds with a snapshot and switches to live delivery."""
    client = create_client(event_log_ws)

    await send_command(client, EVENT_STREAM_SUBSCRIBE, "m1")
    messages = drain_wire(client)
    result = _result(messages[0])
    assert result["mode"] == "snapshot"
    assert result["reason"] == "fresh"
    assert result["server_id"] == mass_minimal.server_id
    assert result["cursor"] == 0
    assert result["token"]
    snapshot = result["snapshot"]
    assert snapshot["players"] == [{"player_id": "p1"}]
    assert snapshot["queue_items"] == {"q1": [{"queue_id": "q1", "queue_item_id": "q1-item"}]}
    assert _envelopes(messages) == []

    # live events after the snapshot arrive as envelopes and advance the cursor
    mass_minimal.signal_event(EventType.PLAYER_UPDATED, "p1", {"name": "Player"})
    await settle()
    envelopes = _envelopes(drain_wire(client))
    assert [message["seq"] for message in envelopes] == [1]
    assert envelopes[0]["data"] == {"name": "Player"}


async def test_reconnect_replays_without_gap_or_duplicate(
    event_log_ws: dict[str, Any],
    mass_minimal: MusicAssistant,
) -> None:
    """Events after the last cursor are replayed on reconnect, response first."""
    client = create_client(event_log_ws)
    await send_command(client, EVENT_STREAM_SUBSCRIBE, "m1")
    token = _result(drain_wire(client)[0])["token"]

    mass_minimal.signal_event(EventType.QUEUE_UPDATED, "q1", {"state": "playing"})
    await settle()
    seq = _envelopes(drain_wire(client))[0]["seq"]

    # the old connection drops (events stay in the global window)
    assert client._event_sub is not None
    mass_minimal.event_log.unregister_listener(client._event_sub)
    mass_minimal.signal_event(EventType.QUEUE_ITEMS_UPDATED, "q1", {"items": []})
    await settle()

    # a new connection presents the token and last cursor
    client2 = create_client(event_log_ws)
    await send_command(
        client2,
        EVENT_STREAM_SUBSCRIBE,
        "m2",
        token=token,
        seq=seq,
        server_id=mass_minimal.server_id,
    )
    messages = drain_wire(client2)
    result = _result(messages[0])
    assert result["mode"] == "resumed"
    assert result["cursor"] == 2
    envelopes = _envelopes(messages[1:])
    assert [message["seq"] for message in envelopes] == [2]
    assert envelopes[0]["event"] == EventType.QUEUE_ITEMS_UPDATED.value

    # events admitted afterwards keep the stream contiguous
    mass_minimal.signal_event(EventType.QUEUE_TIME_UPDATED, "q1", {"elapsed_time": 1})
    await settle()
    assert [m["seq"] for m in _envelopes(drain_wire(client2))] == [3]


async def test_filtered_events_still_advance_the_watermark(
    event_log_ws: dict[str, Any],
    mass_minimal: MusicAssistant,
) -> None:
    """Events hidden from a restricted user create no gap in that user's stream."""
    user = User(
        user_id="restricted",
        username="restricted",
        role=UserRole.USER,
        player_filter=["allowed"],
    )
    client = create_client(event_log_ws, user)
    await send_command(client, EVENT_STREAM_SUBSCRIBE, "m1")
    drain_wire(client)

    mass_minimal.signal_event(EventType.PLAYER_UPDATED, "hidden-player", {})
    mass_minimal.signal_event(EventType.PLAYER_UPDATED, "allowed", {"name": "ok"})
    await settle()

    envelopes = _envelopes(drain_wire(client))
    # only the allowed event is sent, but its seq is 2 - no gap error closed the stream
    assert [message["seq"] for message in envelopes] == [2]
    assert client._event_sub is not None
    assert client._event_sub.sent_watermark == 2


async def test_ack_is_clamped_to_the_contiguous_sent_watermark(
    event_log_ws: dict[str, Any],
    mass_minimal: MusicAssistant,
) -> None:
    """Future acks clamp to the delivered cursor; repeated/lower acks are idempotent."""
    client = create_client(event_log_ws)
    await send_command(client, EVENT_STREAM_SUBSCRIBE, "m1")
    drain_wire(client)
    mass_minimal.signal_event(EventType.PLAYER_UPDATED, "p1")
    await settle()
    drain_wire(client)

    await send_command(client, EVENT_STREAM_ACK, "a1", seq=999)
    assert _result(drain_wire(client)[0])["acked"] == 1

    await send_command(client, EVENT_STREAM_ACK, "a2", seq=1)
    assert _result(drain_wire(client)[0])["acked"] == 1


async def test_snapshot_invalid_cursor_reasons(
    event_log_ws: dict[str, Any],
    mass_minimal: MusicAssistant,
) -> None:
    """Foreign instance, unknown token, future cursor and trimmed window all rebuild."""
    first = create_client(event_log_ws)
    await send_command(first, EVENT_STREAM_SUBSCRIBE, "m1")
    first_result = _result(drain_wire(first)[0])

    async def reopen(message_id: str, **kwargs: Any) -> str:
        client = create_client(event_log_ws)
        await send_command(client, EVENT_STREAM_SUBSCRIBE, message_id, **kwargs)
        reason = _result(drain_wire(client)[0])["reason"]
        return cast("str", reason)

    # unknown token
    assert (
        await reopen("r1", token="does-not-exist", seq=1, server_id=mass_minimal.server_id)
        == "invalid_cursor"
    )
    # token from a different server instance
    assert (
        await reopen("r2", token=first_result["token"], seq=1, server_id="another-server")
        == "invalid_cursor"
    )
    # no token at all is a plain fresh subscription
    assert await reopen("r3", token=None, seq=999, server_id=mass_minimal.server_id) == "fresh"

    # trimmed window: forge a record far below the current window and present a valid
    # token bound to this user's current permission signature
    event_log: EventLogController = event_log_ws["event_log"]
    old = EventRecord(
        seq=10, event="player_updated", object_id="x", ts=0.0, data_type="raw", data=None
    )
    event_log._records.clear()
    event_log._records.append(old)
    event_log._head_seq = 12
    admin_user = first._authenticated_user
    assert admin_user is not None
    token = await event_log.create_subscription(admin_user, compute_scope_sig(admin_user))
    assert await reopen("r4", token=token, seq=5, server_id=mass_minimal.server_id) == "trimmed"


async def test_scope_change_forces_snapshot(
    event_log_ws: dict[str, Any],
    mass_minimal: MusicAssistant,
) -> None:
    """A token presented under another user/permission boundary is not resumed."""
    user_a = User(user_id="a", username="a", role=UserRole.ADMIN)
    user_b = User(user_id="b", username="b", role=UserRole.GUEST)
    client_a = create_client(event_log_ws, user_a)
    await send_command(client_a, EVENT_STREAM_SUBSCRIBE, "m1")
    token = _result(drain_wire(client_a)[0])["token"]

    client_b = create_client(event_log_ws, user_b)
    await send_command(
        client_b,
        EVENT_STREAM_SUBSCRIBE,
        "m2",
        token=token,
        # a positive cursor is required for the token/scope checks to be evaluated;
        # the scope mismatch is detected before the cursor range check
        seq=1,
        server_id=mass_minimal.server_id,
    )
    result = _result(drain_wire(client_b)[0])
    assert result["mode"] == "snapshot"
    assert result["reason"] == "scope_changed"


async def test_events_during_subscribe_are_buffered_behind_response(
    event_log_ws: dict[str, Any],
    mass_minimal: MusicAssistant,
) -> None:
    """No gap: events admitted while the snapshot is gathered land after its response."""
    client = create_client(event_log_ws)
    # kick off the subscribe: listener registration runs eagerly before the first
    # await, so the event signalled immediately afterwards lands in the buffer
    msg = CommandMessage(message_id="m1", command=EVENT_STREAM_SUBSCRIBE, args=None)
    await client._handle_event_stream_command(msg)
    mass_minimal.signal_event(EventType.PLAYER_UPDATED, "p1")
    # wait deterministically for the response, then drain the fan-out callbacks
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5
    needle = '"message_id":"m1"'
    queued_items: Any = client._to_write._queue  # type: ignore[attr-defined]
    while loop.time() < deadline:
        if any(isinstance(item, str) and needle in item for item in queued_items):
            break
        await asyncio.sleep(0)
    await settle()

    messages = drain_wire(client)
    assert "result" in messages[0]
    envelopes = _envelopes(messages)
    assert [message["seq"] for message in envelopes] == [1]


async def test_legacy_client_without_opt_in_is_rejected_and_keeps_bare_events(
    event_log_ws: dict[str, Any],
    mass_minimal: MusicAssistant,
) -> None:
    """Old clients cannot open a stream and still receive plain MassEvent JSON."""
    client = create_client(event_log_ws, opt_in=False)
    client._subscribe_to_events()

    await send_command(client, EVENT_STREAM_SUBSCRIBE, "m1")
    error = drain_wire(client)[0]
    assert "error_code" in error

    mass_minimal.signal_event(EventType.PLAYER_UPDATED, "p1", {"a": 1})
    await settle()
    raw = client._to_write.get_nowait()
    assert raw is not None
    message = json.loads(raw)
    assert message["event"] == EventType.PLAYER_UPDATED.value
    assert "seq" not in message
