"""Retention cleanup safety tests."""

from __future__ import annotations

import time

from .conftest import make_player, wait_for_terminal


async def _count(mass: object, table: str, operation_id: str) -> int:
    rows = await mass.player_commands.database.get_rows_from_query(
        f"SELECT COUNT(*) AS cnt FROM {table} WHERE operation_id = :op",
        {"op": operation_id},
        limit=1,
    )
    return int(rows[0]["cnt"])


async def test_cleanup_cascades_all_rows_for_old_terminal_operation(
    pc_mass: object,
) -> None:
    mass = pc_mass
    make_player(mass, "p1")
    await mass.player_commands.submit(
        operation_id="ret-old",
        command="players/cmd/pause",
        player_ids=["p1"],
    )
    await wait_for_terminal(mass, "ret-old")
    # backdate it beyond retention + event grace
    await mass.player_commands.database.execute_write(
        "UPDATE batch_operations SET finished_at = :ts WHERE operation_id = 'ret-old'",
        {"ts": time.time() - 10 * 86400},
    )
    removed = await mass.player_commands.db.cleanup_terminal_operations(
        cutoff=time.time() - 100
    )
    assert removed == 1
    for table in ("batch_operations", "batch_targets", "batch_events", "batch_audit"):
        assert await _count(mass, table, "ret-old") == 0
    # idempotent: a second sweep has nothing to do
    assert (
        await mass.player_commands.db.cleanup_terminal_operations(
            cutoff=time.time() - 100
        )
        == 0
    )


async def test_young_terminal_operation_is_kept(pc_mass: object) -> None:
    mass = pc_mass
    make_player(mass, "p1")
    await mass.player_commands.submit(
        operation_id="ret-young",
        command="players/cmd/pause",
        player_ids=["p1"],
    )
    await wait_for_terminal(mass, "ret-young")
    removed = await mass.player_commands.db.cleanup_terminal_operations(
        cutoff=time.time() - 86400
    )
    assert removed == 0
    assert await _count(mass, "batch_operations", "ret-young") == 1


async def test_active_operation_is_never_deleted(pc_mass: object) -> None:
    mass = pc_mass
    await mass.player_commands.submit(
        operation_id="ret-active",
        command="players/cmd/pause",
        player_ids=["ghost"],
    )
    removed = await mass.player_commands.db.cleanup_terminal_operations(
        cutoff=time.time() + 86400
    )
    assert removed == 0
    assert await _count(mass, "batch_operations", "ret-active") == 1


async def test_event_grace_extends_operation_lifetime(pc_mass: object) -> None:
    mass = pc_mass
    make_player(mass, "p1")
    await mass.player_commands.submit(
        operation_id="ret-grace",
        command="players/cmd/pause",
        player_ids=["p1"],
    )
    await wait_for_terminal(mass, "ret-grace")
    # finished 30 hours ago: beyond the 1 day retention but inside the +24h event grace
    finished_at = time.time() - 30 * 3600
    await mass.player_commands.database.execute_write(
        "UPDATE batch_operations SET finished_at = :ts WHERE operation_id = 'ret-grace'",
        {"ts": finished_at},
    )
    # retention 1 day (cutoff now-1d): without grace this would be deleted, but the
    # handler's cutoff also includes the configured 24h event grace
    mass.config.set_raw_core_config_value("player_commands", "retention_days", 1)
    mass.config.set_raw_core_config_value("player_commands", "event_grace_hours", 24)
    await mass.player_commands._cleanup_expired()
    assert await _count(mass, "batch_operations", "ret-grace") == 1
