"""Topology freeze and change reporting tests."""

from __future__ import annotations

from music_assistant_models.errors import PlayerUnavailableError

from music_assistant.helpers.json import json_loads

from .conftest import (
    install_fake_command,
    make_player,
    target_map,
    wait_for_target_status,
    wait_for_terminal,
)


async def test_temporary_unregister_then_return_succeeds(pc_mass: object) -> None:
    """A temporarily absent player is retried and succeeds on its return."""
    mass = pc_mass
    make_player(mass, "roamer", provider_instance="prov-1")
    entered = False

    async def dispatcher(_player_id: str) -> None:
        nonlocal entered
        entered = True

    install_fake_command(mass, "players/cmd/pause", dispatcher)
    await mass.player_commands.submit(
        operation_id="topo-return",
        command="players/cmd/pause",
        player_ids=["roamer"],
    )
    await wait_for_terminal(mass, "topo-return")
    assert entered is True
    # second operation: player goes away temporarily and comes back
    entered2 = False

    async def dispatcher2(_player_id: str) -> None:
        nonlocal entered2
        entered2 = True

    install_fake_command(mass, "players/cmd/pause", dispatcher2)
    mass.players._players.pop("roamer")
    mass.player_commands.notify_player_gone("roamer", permanent=False)
    await mass.player_commands.submit(
        operation_id="topo-return-2",
        command="players/cmd/pause",
        player_ids=["roamer"],
    )
    await asyncio_pending(mass, "topo-return-2", "roamer")
    make_player(mass, "roamer", provider_instance="prov-1")
    mass.player_commands.notify_player_available("roamer", "prov-1", "test_provider")
    row = await wait_for_terminal(mass, "topo-return-2")
    assert row["status"] == "succeeded"
    assert entered2 is True


async def asyncio_pending(mass: object, operation_id: str, player_id: str) -> None:
    """Wait until a target is present in the pending state."""
    await wait_for_target_status(mass, operation_id, player_id, "pending")


async def test_permanent_unregister_queued_is_unreachable(pc_mass: object) -> None:
    """A permanently removed pending target becomes unreachable."""
    mass = pc_mass
    # never-resolved target stays pending; a permanent removal settles it
    await mass.player_commands.submit(
        operation_id="topo-gone",
        command="players/cmd/pause",
        player_ids=["ghost"],
    )
    await wait_for_target_status(mass, "topo-gone", "ghost", "pending")
    mass.player_commands.notify_player_gone("ghost", permanent=True)
    row = await wait_for_terminal(mass, "topo-gone")
    assert row["status"] == "failed"
    target = target_map(await mass.player_commands.build_snapshot(row))["ghost"]
    assert target.status == "unreachable"
    assert target.last_error_code == 1001


async def test_permanent_unregister_in_flight_success_stands(pc_mass: object) -> None:
    """An in-flight success confirmed during removal stays succeeded."""
    mass = pc_mass
    make_player(mass, "p1", provider_instance="prov-1")

    async def succeed_while_removed(_player_id: str) -> None:
        # the device confirmed the command while the registry is dropping the player
        mass.players._players.pop("p1", None)
        mass.player_commands.notify_player_gone("p1", permanent=True)

    install_fake_command(mass, "players/cmd/pause", succeed_while_removed)
    await mass.player_commands.submit(
        operation_id="topo-inflight-ok",
        command="players/cmd/pause",
        player_ids=["p1"],
    )
    row = await wait_for_terminal(mass, "topo-inflight-ok")
    assert row["status"] == "succeeded"
    target = target_map(await mass.player_commands.build_snapshot(row))["p1"]
    assert target.status == "succeeded"


async def test_permanent_unregister_in_flight_error_is_unreachable(
    pc_mass: object,
) -> None:
    """An in-flight error during permanent removal becomes unreachable."""
    mass = pc_mass
    make_player(mass, "p1", provider_instance="prov-1")

    async def fail_while_removed(_player_id: str) -> None:
        mass.players._players.pop("p1", None)
        mass.player_commands.notify_player_gone("p1", permanent=True)
        raise PlayerUnavailableError("gone")

    install_fake_command(mass, "players/cmd/pause", fail_while_removed)
    await mass.player_commands.submit(
        operation_id="topo-inflight-err",
        command="players/cmd/pause",
        player_ids=["p1"],
    )
    row = await wait_for_terminal(mass, "topo-inflight-err")
    target = target_map(await mass.player_commands.build_snapshot(row))["p1"]
    assert target.status == "unreachable"
    assert target.attempts == 1


async def test_replaced_provider_instance_queued_is_replaced(pc_mass: object) -> None:
    """A queued target on a replaced provider identity becomes replaced."""
    mass = pc_mass
    player = make_player(mass, "p1", provider_instance="inst-a")
    # freeze the old identity but keep the target un-admitted while we swap identities
    player.initialized.clear()
    await mass.player_commands.submit(
        operation_id="topo-replaced",
        command="players/cmd/pause",
        player_ids=["p1"],
    )
    await wait_for_target_status(mass, "topo-replaced", "p1", "pending")
    mass.player_commands.notify_player_replaced("p1", "inst-a", "test_provider")
    row = await wait_for_terminal(mass, "topo-replaced")
    target = target_map(await mass.player_commands.build_snapshot(row))["p1"]
    assert target.status == "replaced"
    assert target.last_error_code == 1002


async def test_replaced_midflight_result_is_annotated_not_flipped(pc_mass: object) -> None:
    """Replacement during an in-flight command only annotates the success event."""
    mass = pc_mass
    make_player(mass, "p1", provider_instance="inst-a")

    async def succeed_while_replaced(_player_id: str) -> None:
        replacement = make_player(mass, "p1", provider_instance="inst-b")
        mass.player_commands.notify_player_replaced("p1", "inst-a", "test_provider")
        assert replacement.player_id == "p1"

    install_fake_command(mass, "players/cmd/pause", succeed_while_replaced)
    await mass.player_commands.submit(
        operation_id="topo-annotate",
        command="players/cmd/pause",
        player_ids=["p1"],
    )
    row = await wait_for_terminal(mass, "topo-annotate")
    assert row["status"] == "succeeded"
    target = target_map(await mass.player_commands.build_snapshot(row))["p1"]
    assert target.status == "succeeded"
    events = await mass.player_commands.db.get_event_rows(
        after_cursor=0, operation_id="topo-annotate", limit=100
    )
    succeeded = [event for event in events if event["event_type"] == "target_succeeded"]
    assert any(
        json_loads(event["payload_json"]).get("replaced_after_dispatch") is True
        for event in succeeded
    )


async def test_new_players_do_not_expand_frozen_target_set(pc_mass: object) -> None:
    """Players registering after submit never join a frozen target set."""
    mass = pc_mass
    make_player(mass, "p1")
    result = await mass.player_commands.submit(
        operation_id="topo-freeze",
        command="players/cmd/pause",
        player_ids=["p1"],
    )
    frozen = list(target_map(result.snapshot))
    make_player(mass, "p2-late")
    mass.player_commands.notify_player_available(
        "p2-late", "inst-new", "test_provider"
    )
    await wait_for_terminal(mass, "topo-freeze")
    snap = await mass.player_commands.get_operation("topo-freeze")
    assert [target.player_id for target in snap.targets] == frozen == ["p1"]
