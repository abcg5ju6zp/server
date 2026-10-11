"""Cursor subscription, resume, gap resync and dynamic permission filtering."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import Token

import pytest
from music_assistant_models.auth import User, UserRole
from music_assistant_models.errors import PlayerUnavailableError

from music_assistant.controllers.webserver.helpers.auth_middleware import (
    current_user,
)

from .conftest import (
    install_fake_command,
    make_player,
    wait_for_terminal,
)


@contextmanager
def as_user(user: User | None) -> Token:
    """Temporarily run the block as the given user."""
    token = current_user.set(user)
    try:
        yield token
    finally:
        current_user.reset(token)


async def _collect(mass: object, operation_id: str, **kwargs: object) -> object:
    gen = mass.player_commands.subscribe_events(operation_id=operation_id, **kwargs)
    try:
        return await gen.__anext__()
    finally:
        await gen.aclose()


async def test_subscribe_returns_events_and_advances_cursor(pc_mass: object) -> None:
    """Subscription returns ordered events and a monotonic next cursor."""
    mass = pc_mass

    async def ok(_player_id: str) -> None:
        return None

    make_player(mass, "p1")
    install_fake_command(mass, "players/cmd/pause", ok)
    await mass.player_commands.submit(
        operation_id="sub-1",
        command="players/cmd/pause",
        player_ids=["p1"],
    )
    await wait_for_terminal(mass, "sub-1")
    page = await _collect(mass, "sub-1", after_cursor=0, limit=200, wait_timeout=5)
    types = [event.event_type for event in page.events]
    assert "operation_created" in types
    assert "target_dispatched" in types
    assert "target_succeeded" in types
    assert "operation_finished" in types
    seqs = [event.cursor for event in page.events]
    assert seqs == sorted(seqs)
    assert page.next_cursor == seqs[-1]
    assert page.terminal is True
    # resuming past the end returns an empty terminal page, not the same events
    tail = await _collect(
        mass, "sub-1", after_cursor=page.next_cursor, limit=200, wait_timeout=0.1
    )
    assert tail.events == []
    assert tail.terminal is True
    assert tail.next_cursor == page.next_cursor


async def test_resume_from_cursor_no_loss_or_duplication(pc_mass: object) -> None:
    """Paged resumption from a cursor loses and duplicates no events."""
    mass = pc_mass

    async def ok(_player_id: str) -> None:
        return None

    for pid in ("p1", "p2", "p3"):
        make_player(mass, pid)
    install_fake_command(mass, "players/cmd/pause", ok)
    await mass.player_commands.submit(
        operation_id="sub-page",
        command="players/cmd/pause",
        player_ids=["p1", "p2", "p3"],
    )
    await wait_for_terminal(mass, "sub-page")
    seen: list[int] = []
    cursor = 0
    while True:
        page = await _collect(
            mass, "sub-page", after_cursor=cursor, limit=2, wait_timeout=5
        )
        seen.extend(event.cursor for event in page.events)
        cursor = page.next_cursor
        if page.terminal:
            break
    assert len(seen) == len(set(seen))
    db_seqs = [
        int(row["seq"])
        for row in await mass.player_commands.db.get_event_rows(
            after_cursor=0, operation_id="sub-page", limit=1000
        )
    ]
    assert seen == db_seqs


async def test_gap_returns_resync_with_snapshot(pc_mass: object) -> None:
    """A cursor older than the retained window requests resync with a snapshot."""
    mass = pc_mass

    async def ok(_player_id: str) -> None:
        return None

    make_player(mass, "p1")
    install_fake_command(mass, "players/cmd/pause", ok)
    await mass.player_commands.submit(
        operation_id="sub-gap",
        command="players/cmd/pause",
        player_ids=["p1"],
    )
    await wait_for_terminal(mass, "sub-gap")
    # simulate retention pruning the early window while a slow client kept an old cursor
    await mass.player_commands.database.execute_write(
        "DELETE FROM batch_events WHERE operation_id = 'sub-gap' AND seq <= "
        "(SELECT MIN(seq) + 1 FROM batch_events WHERE operation_id = 'sub-gap')"
    )
    page = await _collect(mass, "sub-gap", after_cursor=1, limit=200, wait_timeout=1)
    assert page.resync is True
    assert page.resync_reason == "gap"
    assert page.snapshot is not None
    assert page.snapshot.operation.operation_id == "sub-gap"


async def test_deleted_operation_returns_operation_gone(pc_mass: object) -> None:
    """Subscribing to a missing operation returns an operation_gone resync."""
    page = await _collect(pc_mass, "no-such-op", after_cursor=0, wait_timeout=0.1)
    assert page.resync is True
    assert page.resync_reason == "operation_gone"
    assert page.terminal is True


async def test_events_filtered_after_access_revoked(pc_mass: object) -> None:
    """Target events are filtered dynamically by the caller's current player access."""
    mass = pc_mass

    async def ok(_player_id: str) -> None:
        return None

    make_player(mass, "p1")
    make_player(mass, "p2")
    install_fake_command(mass, "players/cmd/pause", ok)
    await mass.player_commands.submit(
        operation_id="sub-filter",
        command="players/cmd/pause",
        player_ids=["p1", "p2"],
    )
    await wait_for_terminal(mass, "sub-filter")
    restricted = User(
        user_id="restricted-user",
        username="restricted",
        role=UserRole.USER,
        player_filter=["p2"],
    )
    with as_user(restricted):
        page = await _collect(mass, "sub-filter", after_cursor=0, limit=200, wait_timeout=1)
    player_events = [event for event in page.events if event.player_id is not None]
    assert {event.player_id for event in player_events} == {"p2"}
    # operation-level events remain visible because p2 is still part of the operation
    assert any(event.player_id is None for event in page.events)
    excluded = User(
        user_id="excluded-user",
        username="excluded",
        role=UserRole.USER,
        player_filter=["p9"],
    )
    with as_user(excluded):
        hidden = await _collect(mass, "sub-filter", after_cursor=0, limit=200, wait_timeout=1)
    assert hidden.resync is True
    assert hidden.resync_reason == "operation_gone"


async def test_snapshot_hides_targets_lost_after_submit(pc_mass: object) -> None:
    """Losing access after submit hides the operation from later reads."""
    mass = pc_mass

    async def ok(_player_id: str) -> None:
        return None

    make_player(mass, "p1")
    install_fake_command(mass, "players/cmd/pause", ok)
    await mass.player_commands.submit(
        operation_id="sub-view",
        command="players/cmd/pause",
        player_ids=["p1"],
    )
    await wait_for_terminal(mass, "sub-view")
    restricted = User(
        user_id="view-user",
        username="view",
        role=UserRole.USER,
        player_filter=["other"],
    )
    with as_user(restricted), pytest.raises(PlayerUnavailableError):
        await mass.player_commands.get_operation("sub-view")
