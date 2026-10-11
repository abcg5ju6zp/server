"""Partial success, concurrency limits, fair scheduling and redirect reporting."""

from __future__ import annotations

import asyncio

from music_assistant_models.enums import PlaybackState

from music_assistant.models.player import Player

from .conftest import (
    install_fake_command,
    make_player,
    target_map,
    wait_for_target_status,
    wait_for_terminal,
)


def per_player(behaviors: dict[str, object]) -> object:
    """Build a dispatcher that routes each player id to its own behavior coroutine."""

    async def dispatcher(player_id: str) -> None:
        behavior = behaviors[player_id]
        result = behavior()
        if asyncio.iscoroutine(result):
            await result

    return dispatcher


async def test_partial_success_aggregation(pc_mass: object) -> None:
    """Mixed target results aggregate the operation to partial."""
    mass = pc_mass

    async def ok() -> None:
        return None

    async def boom() -> None:
        raise RuntimeError("device exploded")

    make_player(mass, "ok1")
    make_player(mass, "ok2")
    make_player(mass, "bad")
    install_fake_command(mass, "players/cmd/pause", per_player(
        {"ok1": ok, "ok2": ok, "bad": boom}
    ))
    result = await mass.player_commands.submit(
        operation_id="partial-1",
        command="players/cmd/pause",
        player_ids=["ok1", "ok2", "bad"],
    )
    row = await wait_for_terminal(mass, "partial-1")
    assert row["status"] == "partial"
    snap = await mass.player_commands.build_snapshot(row)
    targets = target_map(snap)
    assert targets["ok1"].status == targets["ok2"].status == "succeeded"
    assert targets["bad"].status == "failed"
    assert snap.operation.counts.succeeded == 2
    assert snap.operation.counts.failed == 1
    assert result.snapshot.operation.status in ("queued", "running")


async def test_global_concurrency_cap_is_enforced(pc_mass: object) -> None:
    """The global concurrency budget is never exceeded."""
    mass = pc_mass
    mass.config.set_raw_core_config_value(
        "player_commands", "max_global_concurrency", 2
    )
    peak = 0
    current = 0
    lock = asyncio.Lock()
    gate = asyncio.Event()

    async def slow() -> None:
        nonlocal peak, current
        async with lock:
            current += 1
            peak = max(peak, current)
        await gate.wait()
        async with lock:
            current -= 1

    for pid in ("s1", "s2", "s3", "s4"):
        make_player(mass, pid)
    install_fake_command(
        mass, "players/cmd/pause", per_player(dict.fromkeys(("s1", "s2", "s3", "s4"), slow))
    )
    await mass.player_commands.submit(
        operation_id="cap-global",
        command="players/cmd/pause",
        player_ids=["s1", "s2", "s3", "s4"],
    )
    # wait until the first two are in flight, the cap must never have been exceeded
    await wait_for_target_status(mass, "cap-global", "s1", "running")
    await asyncio.sleep(0.15)
    assert peak == 2
    gate.set()
    row = await wait_for_terminal(mass, "cap-global")
    assert row["status"] == "succeeded"
    assert peak == 2


async def test_per_operation_concurrency_cap(pc_mass: object) -> None:
    """The per-operation concurrency limit is respected."""
    mass = pc_mass
    peak = 0
    current = 0
    lock = asyncio.Lock()
    gate = asyncio.Event()

    async def slow() -> None:
        nonlocal peak, current
        async with lock:
            current += 1
            peak = max(peak, current)
        await gate.wait()
        async with lock:
            current -= 1

    for pid in ("o1", "o2", "o3"):
        make_player(mass, pid)
    install_fake_command(
        mass, "players/cmd/pause", per_player(dict.fromkeys(("o1", "o2", "o3"), slow))
    )
    await mass.player_commands.submit(
        operation_id="cap-op",
        command="players/cmd/pause",
        player_ids=["o1", "o2", "o3"],
        max_concurrency=1,
    )
    await wait_for_target_status(mass, "cap-op", "o1", "running")
    await asyncio.sleep(0.1)
    assert peak == 1
    gate.set()
    row = await wait_for_terminal(mass, "cap-op")
    assert row["status"] == "succeeded"


async def test_fair_scheduling_round_robins_across_operations(pc_mass: object) -> None:
    """Dispatch round-robins across operations so no operation starves."""
    mass = pc_mass
    gate = asyncio.Event()
    dispatched: list[str] = []

    for pid in ("a1", "a2", "a3", "a4", "b1", "b2"):
        make_player(mass, pid)

    async def dispatcher(player_id: str) -> None:
        dispatched.append(player_id)
        await gate.wait()

    install_fake_command(mass, "players/cmd/pause", dispatcher)
    await mass.player_commands.submit(
        operation_id="fair-a",
        command="players/cmd/pause",
        player_ids=["a1", "a2", "a3", "a4"],
        max_concurrency=4,
    )
    await mass.player_commands.submit(
        operation_id="fair-b",
        command="players/cmd/pause",
        player_ids=["b1", "b2"],
        max_concurrency=2,
    )
    await wait_for_target_status(mass, "fair-b", "b1", "running")
    await asyncio.sleep(0.05)
    # the second operation's first target must be admitted before the first operation
    # takes every global slot
    order_before_release = list(dispatched)
    assert "b1" in order_before_release
    assert order_before_release.index("b1") < 4
    gate.set()
    await wait_for_terminal(mass, "fair-a")
    await wait_for_terminal(mass, "fair-b")


async def test_redirected_target_records_effective_player(pc_mass: object) -> None:
    """A grouped/synced target records redirect outcome and effective player."""
    mass = pc_mass
    leader = make_player(mass, "leader", playback_state=PlaybackState.PLAYING)
    member = make_player(mass, "member", playback_state=PlaybackState.PLAYING)
    # mirror a sync group: the member's playback is owned by the leader
    member.state.synced_to = "leader"
    leader.state.group_members = ["leader", "member"]
    called_on: list[str] = []

    async def dispatcher(player_id: str) -> None:
        called_on.append(player_id)

    install_fake_command(mass, "players/cmd/pause", dispatcher)
    await mass.player_commands.submit(
        operation_id="redirect-1",
        command="players/cmd/pause",
        player_ids=["member"],
    )
    row = await wait_for_terminal(mass, "redirect-1")
    assert row["status"] == "succeeded"
    targets = target_map(await mass.player_commands.build_snapshot(row))
    target = targets["member"]
    assert target.status == "succeeded"
    assert target.outcome == "redirected"
    assert target.effective_player_id == "leader"
    assert isinstance(leader, Player)
