"""Permission filtering at creation, read-back and execution time."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import Token

import pytest
from music_assistant_models.auth import User, UserRole
from music_assistant_models.errors import (
    InsufficientPermissions,
    PlayerUnavailableError,
)

from music_assistant.controllers.webserver.helpers.auth_middleware import (
    current_user,
    get_current_user,
)

from .conftest import (
    install_fake_command,
    make_player,
    target_map,
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


def restricted_user(
    user_id: str = "limited", username: str = "limited", allowed: list[str] | None = None
) -> User:
    """Build an ephemeral member user limited to the given player ids."""
    return User(
        user_id=user_id,
        username=username,
        role=UserRole.USER,
        player_filter=allowed or [],
    )


async def test_invisible_target_terminal_others_execute(pc_mass: object) -> None:
    """An invisible player fails per-target while visible targets execute."""
    mass = pc_mass

    async def ok(_player_id: str) -> None:
        return None

    make_player(mass, "visible")
    make_player(mass, "secret")
    install_fake_command(mass, "players/cmd/pause", ok)
    user = restricted_user(allowed=["visible"])
    with as_user(user):
        result = await mass.player_commands.submit(
            operation_id="perm-create",
            command="players/cmd/pause",
            player_ids=["visible", "secret"],
        )
    targets = target_map(result.snapshot)
    assert targets["visible"].status == "pending"
    assert targets["secret"].status == "failed"
    assert targets["secret"].last_error_code == 22
    row = await wait_for_terminal(mass, "perm-create")
    assert row["status"] == "partial"


async def test_snapshot_and_events_filter_for_current_user(pc_mass: object) -> None:
    """Snapshots are filtered to the targets the requesting user may see."""
    mass = pc_mass

    async def ok(_player_id: str) -> None:
        return None

    make_player(mass, "p1")
    make_player(mass, "p2")
    install_fake_command(mass, "players/cmd/pause", ok)
    await mass.player_commands.submit(
        operation_id="perm-view",
        command="players/cmd/pause",
        player_ids=["p1", "p2"],
    )
    await wait_for_terminal(mass, "perm-view")
    user = restricted_user(allowed=["p1"])
    with as_user(user):
        snap = await mass.player_commands.get_operation("perm-view")
    assert [target.player_id for target in snap.targets] == ["p1"]


async def test_get_without_any_visible_target_is_hidden(pc_mass: object) -> None:
    """A get for an operation with no visible target is hidden as unavailable."""
    mass = pc_mass

    async def ok(_player_id: str) -> None:
        return None

    make_player(mass, "p1")
    install_fake_command(mass, "players/cmd/pause", ok)
    await mass.player_commands.submit(
        operation_id="perm-hidden",
        command="players/cmd/pause",
        player_ids=["p1"],
    )
    await wait_for_terminal(mass, "perm-hidden")
    with as_user(restricted_user(allowed=["nope"])), pytest.raises(PlayerUnavailableError):
        await mass.player_commands.get_operation("perm-hidden")


async def test_execution_runs_with_operation_owner_context(pc_mass: object) -> None:
    """Device commands execute with the frozen operation owner in context."""
    mass = pc_mass
    owner = await mass.webserver.auth.create_user(
        username="batchowner", role=UserRole.USER
    )
    seen: list[str | None] = []

    async def capture(_player_id: str) -> None:
        seen.append(get_current_user().user_id if get_current_user() else None)

    make_player(mass, "p1")
    install_fake_command(mass, "players/cmd/pause", capture)
    with as_user(owner):
        await mass.player_commands.submit(
            operation_id="perm-owner",
            command="players/cmd/pause",
            player_ids=["p1"],
        )
    row = await wait_for_terminal(mass, "perm-owner")
    assert row["status"] == "succeeded"
    assert seen == [owner.user_id]


async def test_cancel_denied_for_user_without_any_access(pc_mass: object) -> None:
    """A user without access to any target may not cancel the operation."""
    mass = pc_mass
    owner = await mass.webserver.auth.create_user(
        username="batchowner2", role=UserRole.USER
    )

    async def blocked(_player_id: str) -> None:
        await asyncio.Event().wait()

    make_player(mass, "p1")
    install_fake_command(mass, "players/cmd/pause", blocked)
    with as_user(owner):
        await mass.player_commands.submit(
            operation_id="perm-cancel",
            command="players/cmd/pause",
            player_ids=["p1"],
        )
    with (
        as_user(restricted_user(user_id="other", username="other", allowed=["nope"])),
        pytest.raises(InsufficientPermissions),
    ):
        await mass.player_commands.cancel_operation("perm-cancel")