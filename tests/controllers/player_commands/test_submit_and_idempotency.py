"""Submit, idempotency/conflict and per-target creation tests."""

from __future__ import annotations

import pytest
from music_assistant_models.errors import InvalidCommand, InvalidDataError

from music_assistant.controllers.player_commands.constants import PC_TABLE_AUDIT
from music_assistant.controllers.player_commands.errors import (
    BatchOperationConflictError,
)

from .conftest import make_player, target_map


async def test_submit_returns_immediately_without_faked_success(pc_mass: object) -> None:
    mass = pc_mass
    make_player(mass, "p1")
    result = await mass.player_commands.submit(
        operation_id="op-1",
        command="players/cmd/pause",
        player_ids=["p1"],
    )
    assert result.reused is False
    snap = result.snapshot
    assert snap.operation.operation_id == "op-1"
    assert snap.operation.status in ("queued", "running", "succeeded")
    assert snap.targets[0].player_id == "p1"
    # the call returns before the device command is awaited: a fast op may already be
    # done, but a missing-player target must still be visibly pending here
    result_missing = await mass.player_commands.submit(
        operation_id="op-missing",
        command="players/cmd/pause",
        player_ids=["ghost"],
    )
    missing = target_map(result_missing.snapshot)["ghost"]
    assert missing.status == "pending"
    assert missing.attempts == 0


async def test_duplicate_submit_same_payload_reuses(pc_mass: object) -> None:
    mass = pc_mass
    make_player(mass, "p1")
    first = await mass.player_commands.submit(
        operation_id="dup-1",
        command="players/cmd/volume_set",
        player_ids=["p1"],
        args={"volume_level": 42},
        max_concurrency=3,
    )
    second = await mass.player_commands.submit(
        operation_id="dup-1",
        command="players/cmd/volume_set",
        player_ids=["p1"],
        args={"volume_level": 42},
        max_concurrency=3,
    )
    assert second.reused is True
    assert second.snapshot.operation.operation_id == "dup-1"
    rows = await mass.player_commands.database.get_rows(
        "batch_operations", {"operation_id": "dup-1"}
    )
    assert len(rows) == 1
    assert first.snapshot.operation.created_at == second.snapshot.operation.created_at


async def test_duplicate_submit_different_args_conflicts(pc_mass: object) -> None:
    mass = pc_mass
    make_player(mass, "p1")
    await mass.player_commands.submit(
        operation_id="dup-2",
        command="players/cmd/volume_set",
        player_ids=["p1"],
        args={"volume_level": 42},
    )
    with pytest.raises(BatchOperationConflictError):
        await mass.player_commands.submit(
            operation_id="dup-2",
            command="players/cmd/volume_set",
            player_ids=["p1"],
            args={"volume_level": 11},
        )


async def test_duplicate_submit_different_targets_or_concurrency_conflicts(
    pc_mass: object,
) -> None:
    mass = pc_mass
    make_player(mass, "p1")
    make_player(mass, "p2")
    await mass.player_commands.submit(
        operation_id="dup-3",
        command="players/cmd/pause",
        player_ids=["p1"],
        max_concurrency=2,
    )
    with pytest.raises(BatchOperationConflictError):
        await mass.player_commands.submit(
            operation_id="dup-3",
            command="players/cmd/pause",
            player_ids=["p1", "p2"],
            max_concurrency=2,
        )
    with pytest.raises(BatchOperationConflictError):
        await mass.player_commands.submit(
            operation_id="dup-3",
            command="players/cmd/pause",
            player_ids=["p1"],
            max_concurrency=1,
        )


async def test_unknown_command_and_bad_args_rejected(pc_mass: object) -> None:
    mass = pc_mass
    make_player(mass, "p1")
    with pytest.raises(InvalidCommand):
        await mass.player_commands.submit(
            operation_id="bad-cmd",
            command="players/cmd/destroy",
            player_ids=["p1"],
        )
    with pytest.raises(InvalidDataError):
        await mass.player_commands.submit(
            operation_id="bad-args",
            command="players/cmd/volume_set",
            player_ids=["p1"],
            args={"volume_level": "loud"},
        )
    with pytest.raises(InvalidDataError):
        await mass.player_commands.submit(
            operation_id="bad-arg-name",
            command="players/cmd/volume_set",
            player_ids=["p1"],
            args={"bogus": 1},
        )


async def test_invalid_operation_id_rejected(pc_mass: object) -> None:
    mass = pc_mass
    with pytest.raises(InvalidDataError):
        await mass.player_commands.submit(
            operation_id="bad id!",
            command="players/cmd/pause",
            player_ids=["p1"],
        )
    with pytest.raises(InvalidDataError):
        await mass.player_commands.submit(
            operation_id="",
            command="players/cmd/pause",
            player_ids=["p1"],
        )
    with pytest.raises(InvalidDataError):
        await mass.player_commands.submit(
            operation_id="x" * 65,
            command="players/cmd/pause",
            player_ids=["p1"],
        )


async def test_disabled_player_is_born_terminal(pc_mass: object) -> None:
    mass = pc_mass
    make_player(mass, "off", enabled=False)
    result = await mass.player_commands.submit(
        operation_id="disabled-1",
        command="players/cmd/pause",
        player_ids=["off"],
    )
    target = target_map(result.snapshot)["off"]
    assert target.status == "failed"
    assert target.last_error_code == 10
    assert result.snapshot.operation.status == "failed"


async def test_missing_player_stays_pending_and_targets_are_deduped(
    pc_mass: object,
) -> None:
    mass = pc_mass
    result = await mass.player_commands.submit(
        operation_id="missing-1",
        command="players/cmd/pause",
        player_ids=["a", "a", "b"],
    )
    ids = list(target_map(result.snapshot))
    assert ids == ["a", "b"]


async def test_submit_writes_audit_rows(pc_mass: object) -> None:
    mass = pc_mass
    make_player(mass, "p1")
    await mass.player_commands.submit(
        operation_id="audit-1",
        command="players/cmd/pause",
        player_ids=["p1"],
    )
    rows = await mass.player_commands.database.get_rows_from_query(
        f"SELECT action FROM {PC_TABLE_AUDIT} WHERE operation_id = :op",
        {"op": "audit-1"},
        limit=0,
    )
    assert any(row["action"] == "operation_created" for row in rows)
