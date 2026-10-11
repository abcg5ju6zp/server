"""Cancellation semantics and late-callback (lease generation) races."""

from __future__ import annotations

import asyncio

from music_assistant.controllers.player_commands.constants import TargetStatus
from music_assistant_models.errors import PlayerUnavailableError

from .conftest import (
    install_fake_command,
    make_player,
    target_map,
    wait_for_target_status,
    wait_for_terminal,
)


async def test_cancel_pending_targets_all_cancelled(pc_mass: object) -> None:
    mass = pc_mass
    # target never resolves, so it stays pending and never reaches a device
    result = await mass.player_commands.submit(
        operation_id="cancel-pending",
        command="players/cmd/pause",
        player_ids=["ghost"],
    )
    assert target_map(result.snapshot)["ghost"].status == "pending"
    snap = await mass.player_commands.cancel_operation("cancel-pending")
    assert snap.operation.cancel_requested is True
    assert target_map(snap)["ghost"].status == "cancelled"
    assert snap.operation.status == "cancelled"
    # cancelling an already terminal operation is idempotent
    again = await mass.player_commands.cancel_operation("cancel-pending")
    assert again.operation.status == "cancelled"


async def test_cancel_in_flight_records_boundary_and_success_converges(
    pc_mass: object,
) -> None:
    mass = pc_mass
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow(player_id: str) -> None:
        entered.set()
        await release.wait()

    make_player(mass, "p1")
    install_fake_command(mass, "players/cmd/pause", slow)
    await mass.player_commands.submit(
        operation_id="cancel-running",
        command="players/cmd/pause",
        player_ids=["p1"],
    )
    await wait_for_target_status(mass, "cancel-running", "p1", "running")
    await entered.wait()
    snap = await mass.player_commands.cancel_operation("cancel-running")
    target = target_map(snap)["p1"]
    assert target.cancel_boundary is True
    assert target.status == "running"  # the sent command can not be withdrawn
    event_types = {
        row["event_type"]
        for row in await mass.player_commands.db.get_event_rows(
            after_cursor=0, operation_id="cancel-running", limit=100
        )
    }
    assert "cancel_boundary" in event_types
    release.set()
    row = await wait_for_terminal(mass, "cancel-running")
    # the operation was cancelled but the already-sent command succeeded: the target
    # keeps the real outcome, the aggregate reflects the cancellation request
    assert row["status"] == "cancelled"
    final = target_map(await mass.player_commands.build_snapshot(row))
    assert final["p1"].status == "succeeded"
    assert final["p1"].cancel_boundary is True


async def test_cancel_in_flight_failure_converges_to_cancelled(pc_mass: object) -> None:
    mass = pc_mass
    entered = asyncio.Event()

    async def fail_after_cancel(player_id: str) -> None:
        entered.set()
        # give the cancellation request time to land while the command is in flight
        await asyncio.sleep(0.1)
        raise PlayerUnavailableError("device went away")

    make_player(mass, "p1")
    install_fake_command(mass, "players/cmd/pause", fail_after_cancel)
    await mass.player_commands.submit(
        operation_id="cancel-fail",
        command="players/cmd/pause",
        player_ids=["p1"],
    )
    await wait_for_target_status(mass, "cancel-fail", "p1", "running")
    await entered.wait()
    await mass.player_commands.cancel_operation("cancel-fail")
    row = await wait_for_terminal(mass, "cancel-fail")
    assert row["status"] == "cancelled"
    final = target_map(await mass.player_commands.build_snapshot(row))
    # a failed in-flight command after cancel is not retried (no new dispatch)
    assert final["p1"].status == "cancelled"
    assert final["p1"].attempts == 1
    assert final["p1"].last_error_code == 10


async def test_late_callback_with_stale_generation_is_ignored(pc_mass: object) -> None:
    mass = pc_mass
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow(player_id: str) -> None:
        entered.set()
        await release.wait()

    make_player(mass, "p1")
    install_fake_command(mass, "players/cmd/pause", slow)
    await mass.player_commands.submit(
        operation_id="stale-gen",
        command="players/cmd/pause",
        player_ids=["p1"],
    )
    target = await wait_for_target_status(mass, "stale-gen", "p1", "running")
    stale_generation = int(target["lease_generation"])
    await entered.wait()
    # watchdog expires the lease while the stuck command is still running: a new
    # generation takes over. The stale callback must not flip the converged state.
    applied_stale = await mass.player_commands.db.finish_target(
        target_id=target["id"],
        generation=stale_generation - 1,
        status=TargetStatus.SUCCEEDED,
        now=0,
    )
    assert applied_stale is False
    release.set()
    row = await wait_for_terminal(mass, "stale-gen")
    assert row["status"] == "succeeded"
    final = target_map(await mass.player_commands.build_snapshot(row))
    assert final["p1"].status == "succeeded"
    assert final["p1"].attempts == 1
