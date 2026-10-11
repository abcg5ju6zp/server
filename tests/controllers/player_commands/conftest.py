"""Shared fixtures and helpers for the batch player commands tests."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import PlaybackState, PlayerType

from music_assistant.controllers.player_commands.constants import (
    CONF_PC_COMMAND_TIMEOUT,
    CONF_PC_LEASE_GRACE,
    CONF_PC_MAX_ATTEMPTS,
    CONF_PC_PROVIDER_GRACE,
    CONF_PC_RETRY_BASE,
    CONF_PC_RETRY_CAP,
)
from music_assistant.controllers.player_commands.controller import (
    PlayerCommandsController,
)
from music_assistant.controllers.player_commands.registry import COMMAND_REGISTRY
from tests.common import MockPlayer, MockProvider


def configure_fast_batch(mass: Any) -> None:
    """Configure the batch controller for fast, deterministic test execution."""
    mass.config.set_raw_core_config_value("player_commands", CONF_PC_RETRY_BASE, 0)
    mass.config.set_raw_core_config_value("player_commands", CONF_PC_RETRY_CAP, 0)
    mass.config.set_raw_core_config_value("player_commands", CONF_PC_COMMAND_TIMEOUT, 30)
    mass.config.set_raw_core_config_value("player_commands", CONF_PC_LEASE_GRACE, 60)
    mass.config.set_raw_core_config_value("player_commands", CONF_PC_MAX_ATTEMPTS, 3)
    # do not let the startup seal window interfere with the tests
    mass.config.set_raw_core_config_value(
        "player_commands", CONF_PC_PROVIDER_GRACE, 86400
    )


def make_player(
    mass: Any,
    player_id: str,
    *,
    name: str | None = None,
    provider_instance: str = "test_instance",
    provider_domain: str = "test_provider",
    player_type: PlayerType = PlayerType.PLAYER,
    enabled: bool = True,
    playback_state: PlaybackState = PlaybackState.PLAYING,
) -> MockPlayer:
    """Register a mock player directly in a (real) PlayerController."""
    provider = MockProvider(provider_domain, instance_id=provider_instance, mass=MagicMock())
    player = MockPlayer(provider, player_id, name or player_id, player_type)
    player._attr_playback_state = playback_state
    player.update_state(signal_event=False)
    if not enabled:
        # enabled is normally a config-driven state flag; tests set it directly
        player.state.enabled = False
    mass.players._players[player_id] = player
    player.set_initialized()
    return player


def install_fake_command(
    mass: Any, command: str, dispatcher: Callable[[str], Awaitable[Any]]
) -> None:
    """
    Replace the single-player handler of a registered command on the PlayerController.

    The scheduler resolves handlers with getattr at execution time, so the instance
    attribute shadows the decorated controller method for the duration of the test.
    """
    spec = COMMAND_REGISTRY[command]

    async def _fake(player_id: str, **_kwargs: Any) -> None:
        await dispatcher(player_id)

    setattr(mass.players, spec.handler_attr, _fake)


async def wait_for_terminal(
    mass: Any,
    operation_id: str,
    *,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Wait until the operation reaches a terminal state and return its raw row."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = await mass.player_commands.db.get_operation_row(operation_id)
        assert row is not None
        if str(row["status"]) in {
            "succeeded",
            "partial",
            "failed",
            "cancelled",
        }:
            return dict(row)
        await asyncio.sleep(0.02)
    msg = f"operation {operation_id} did not finish in time"
    raise AssertionError(msg)


async def wait_for_target_status(
    mass: Any,
    operation_id: str,
    player_id: str,
    status: str,
    *,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Wait until one target of an operation reaches the given raw status value."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for target in await mass.player_commands.db.get_target_rows(operation_id):
            if str(target["player_id"]) == player_id and str(target["status"]) == status:
                return dict(target)
        await asyncio.sleep(0.02)
    msg = f"target {player_id} did not reach {status}"
    raise AssertionError(msg)


async def rebuild_pc_controller(mass: Any) -> PlayerCommandsController:
    """
    Rebuild the batch controller over the same database (process crash simulation).

    The PlayerController and storage survive this; the new boot gets a new boot id and
    takes over leases left behind by the old one.
    """
    await mass.player_commands.close()
    controller = PlayerCommandsController(mass)
    mass.player_commands = controller
    config = await mass.config.get_core_config("player_commands")
    await controller.setup(config)
    await controller.post_setup()
    return controller


def target_map(snapshot: Any) -> dict[str, Any]:
    """Return snapshot targets keyed by player id."""
    return {target.player_id: target for target in snapshot.targets}


@pytest.fixture
async def pc_mass(mass: Any) -> AsyncGenerator[Any]:
    """Boot a full server (no auto providers) configured for fast batch execution."""
    configure_fast_batch(mass)
    yield mass
    with contextlib.suppress(Exception):
        pass
