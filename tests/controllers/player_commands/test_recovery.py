"""Startup recovery, lease takeover, sealing and full process restart tests."""

from __future__ import annotations

import asyncio
import pathlib

from music_assistant.controllers.player_commands.constants import PC_TABLE_AUDIT
from tests.conftest import full_mass_context

from .conftest import (
    configure_fast_batch,
    install_fake_command,
    make_player,
    rebuild_pc_controller,
    target_map,
    wait_for_target_status,
    wait_for_terminal,
)


async def test_rebuild_requeues_stale_lease_and_preserves_attempts(
    pc_mass: object,
) -> None:
    mass = pc_mass
    make_player(mass, "p1")
    gate = asyncio.Event()

    async def blocked(player_id: str) -> None:
        await gate.wait()

    install_fake_command(mass, "players/cmd/pause", blocked)
    await mass.player_commands.submit(
        operation_id="recover-1",
        command="players/cmd/pause",
        player_ids=["p1"],
    )
    target = await wait_for_target_status(mass, "recover-1", "p1", "running")
    assert int(target["attempts"]) == 1
    old_owner = target["lease_owner"]

    # simulates a process crash: close cancels the in-flight task, the lease stays
    controller = await rebuild_pc_controller(mass)
    assert controller.lease_owner != old_owner
    # recovery moved the stale lease back to retry_wait (the new scheduler may already
    # have re-admitted it, so assert the recovery event rather than the racy status)
    events = await mass.player_commands.db.get_event_rows(
        after_cursor=0, operation_id="recover-1", limit=100
    )
    recovery_events = [
        event
        for event in events
        if event["event_type"] == "target_retrying"
        and "boot_recovery" in str(event["payload_json"])
    ]
    assert recovery_events
    # the second dispatch is still behind the gate: release it and expect success
    gate.set()
    row = await wait_for_terminal(mass, "recover-1")
    assert row["status"] == "succeeded"
    final = target_map(await mass.player_commands.build_snapshot(row))["p1"]
    assert final.attempts == 2


async def test_succeeded_target_is_never_run_again(pc_mass: object) -> None:
    mass = pc_mass
    make_player(mass, "done")

    async def ok(player_id: str) -> None:
        return None

    install_fake_command(mass, "players/cmd/pause", ok)
    await mass.player_commands.submit(
        operation_id="recover-done",
        command="players/cmd/pause",
        player_ids=["done"],
    )
    row = await wait_for_terminal(mass, "recover-done")
    assert row["status"] == "succeeded"

    async def fail_if_called(player_id: str) -> None:
        raise AssertionError("succeeded target must never be dispatched again")

    install_fake_command(mass, "players/cmd/pause", fail_if_called)
    await rebuild_pc_controller(mass)
    await asyncio.sleep(0.2)
    rows = await mass.player_commands.database.get_rows_from_query(
        f"SELECT COUNT(*) AS cnt FROM {PC_TABLE_AUDIT} "
        "WHERE operation_id = :op AND action = 'target_dispatched' AND player_id = :pid",
        {"op": "recover-done", "pid": "done"},
        limit=1,
    )
    assert int(rows[0]["cnt"]) == 1
    final = target_map(
        await mass.player_commands.build_snapshot(
            await mass.player_commands.db.get_operation_row("recover-done")
        )
    )["done"]
    assert final.status == "succeeded"
    assert final.attempts == 1


async def test_seal_now_marks_never_resolved_targets(pc_mass: object) -> None:
    mass = pc_mass
    await mass.player_commands.submit(
        operation_id="seal-1",
        command="players/cmd/pause",
        player_ids=["ghost-a", "ghost-b"],
    )
    for pid in ("ghost-a", "ghost-b"):
        await wait_for_target_status(mass, "seal-1", pid, "pending")
    await mass.player_commands.seal_now()
    row = await wait_for_terminal(mass, "seal-1")
    assert row["status"] == "failed"
    targets = target_map(await mass.player_commands.build_snapshot(row))
    assert targets["ghost-a"].status == "sealed"
    assert targets["ghost-b"].status == "sealed"


async def test_seal_does_not_touch_targets_that_were_dispatched(pc_mass: object) -> None:
    mass = pc_mass
    # a target in retry_wait with attempts > 0 must not be sealed
    await mass.player_commands.submit(
        operation_id="seal-2",
        command="players/cmd/pause",
        player_ids=["ghost"],
    )
    await wait_for_target_status(mass, "seal-2", "ghost", "pending")
    # seed a dispatched-but-waiting target directly: sealing must leave it alone
    await mass.player_commands.database.execute_write(
        "UPDATE batch_targets SET status = 'retry_wait', attempts = 1, next_run_at = NULL, "
        "lease_owner = NULL WHERE operation_id = 'seal-2'"
    )
    await mass.player_commands.seal_now()
    await asyncio.sleep(0.1)
    targets = target_map(
        await mass.player_commands.build_snapshot(
            await mass.player_commands.db.get_operation_row("seal-2")
        )
    )
    assert targets["ghost"].status == "retry_wait"
    assert targets["ghost"].attempts == 1


async def test_full_restart_same_storage_recovers_running_target(
    tmp_path: pathlib.Path,
) -> None:
    storage = tmp_path / "data"
    cache = tmp_path / "cache"

    async with full_mass_context(tmp_path) as mass1:
        configure_fast_batch(mass1)
        make_player(mass1, "p1")
        gate = asyncio.Event()

        async def blocked(player_id: str) -> None:
            await gate.wait()

        install_fake_command(mass1, "players/cmd/pause", blocked)
        await mass1.player_commands.submit(
            operation_id="restart-1",
            command="players/cmd/pause",
            player_ids=["p1"],
        )
        await wait_for_target_status(mass1, "restart-1", "p1", "running")
        first_owner = mass1.player_commands.lease_owner

    async with full_mass_context(tmp_path) as mass2:
        configure_fast_batch(mass2)

        async def ok(player_id: str) -> None:
            return None

        install_fake_command(mass2, "players/cmd/pause", ok)
        make_player(mass2, "p1")
        mass2.player_commands.notify_player_available(
            "p1", "test_instance", "test_provider"
        )
        row = await wait_for_terminal(mass2, "restart-1")
        assert row["status"] == "succeeded"
        assert mass2.player_commands.lease_owner != first_owner
        rows = await mass2.player_commands.database.get_rows_from_query(
            f"SELECT COUNT(*) AS cnt FROM {PC_TABLE_AUDIT} "
            "WHERE operation_id = :op AND action = 'target_dispatched'",
            {"op": "restart-1"},
            limit=1,
        )
        assert int(rows[0]["cnt"]) == 2
        targets = target_map(await mass2.player_commands.build_snapshot(row))
        assert targets["p1"].attempts == 2

    assert storage.exists()
    assert cache is not None
