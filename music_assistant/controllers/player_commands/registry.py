"""
Registry of single-player commands that can be driven through the batch protocol.

Only commands that address a single player identity with plain JSON-serializable
arguments are registered. Multi-entity topology mutations (set_members/group_many/
ungroup_many, which touch players outside the frozen target set), media playback
commands carrying media items and timer commands are deliberately excluded in v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

from music_assistant.controllers.players.constants import PlayerLockPurpose

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from music_assistant.controllers.players.controller import PlayerController


@dataclass(frozen=True)
class CommandSpec:
    """Specification of one single-player command usable in a batch operation."""

    command: str
    handler_attr: str
    lock_purpose: PlayerLockPurpose
    # arguments (besides player_id) that the command accepts, in handler order
    extra_params: tuple[str, ...] = ()

    def bind(self, players: PlayerController) -> Callable[..., Awaitable[None]]:
        """Return the bound handler coroutine on the PlayerController."""
        return cast("Callable[..., Awaitable[None]]", getattr(players, self.handler_attr))


COMMAND_REGISTRY: Final[dict[str, CommandSpec]] = {
    spec.command: spec
    for spec in (
        CommandSpec("players/cmd/play", "cmd_play", PlayerLockPurpose.PLAYBACK),
        CommandSpec("players/cmd/pause", "cmd_pause", PlayerLockPurpose.PLAYBACK),
        CommandSpec("players/cmd/play_pause", "cmd_play_pause", PlayerLockPurpose.PLAYBACK),
        CommandSpec("players/cmd/stop", "cmd_stop", PlayerLockPurpose.PLAYBACK),
        CommandSpec("players/cmd/next", "cmd_next_track", PlayerLockPurpose.PLAYBACK),
        CommandSpec("players/cmd/previous", "cmd_previous_track", PlayerLockPurpose.PLAYBACK),
        CommandSpec("players/cmd/seek", "cmd_seek", PlayerLockPurpose.PLAYBACK, ("position",)),
        CommandSpec(
            "players/cmd/shuffle",
            "cmd_shuffle",
            PlayerLockPurpose.PLAYBACK,
            ("shuffle_enabled", "source_id"),
        ),
        CommandSpec(
            "players/cmd/repeat",
            "cmd_repeat",
            PlayerLockPurpose.PLAYBACK,
            ("repeat_mode", "source_id"),
        ),
        CommandSpec("players/cmd/power", "cmd_power", PlayerLockPurpose.PLAYBACK, ("powered",)),
        CommandSpec(
            "players/cmd/volume_set",
            "cmd_volume_set",
            PlayerLockPurpose.VOLUME,
            ("volume_level",),
        ),
        CommandSpec(
            "players/cmd/volume_up",
            "cmd_volume_up",
            PlayerLockPurpose.VOLUME,
        ),
        CommandSpec(
            "players/cmd/volume_down",
            "cmd_volume_down",
            PlayerLockPurpose.VOLUME,
        ),
        CommandSpec(
            "players/cmd/volume_mute",
            "cmd_volume_mute",
            PlayerLockPurpose.VOLUME,
            ("muted",),
        ),
        CommandSpec(
            "players/cmd/group",
            "cmd_group",
            PlayerLockPurpose.PLAYBACK,
            ("target_player",),
        ),
        CommandSpec("players/cmd/ungroup", "cmd_ungroup", PlayerLockPurpose.PLAYBACK),
    )
}
