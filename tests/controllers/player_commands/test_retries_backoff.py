"""Retry/backoff classification and per-target attempt budget tests."""

from __future__ import annotations

from music_assistant_models.errors import (
    PlayerCommandFailed,
    PlayerUnavailableError,
    RateLimited,
    ResourceBusyError,
    UnsupportedFeaturedException,
)

from .conftest import (
    install_fake_command,
    make_player,
    target_map,
    wait_for_terminal,
)


class AttemptBehavior:
    """Raise ``error`` on early attempts and succeed from ``succeed_on_attempt`` on."""

    def __init__(self, error: Exception | None = None, succeed_on_attempt: int = 1) -> None:
        """Configure the scripted error and the attempt at which success starts."""
        self.error = error
        self.succeed_on_attempt = succeed_on_attempt
        self.calls = 0

    async def __call__(self, _player_id: str) -> None:
        """Fail until the configured attempt, then succeed."""
        self.calls += 1
        if self.calls >= self.succeed_on_attempt:
            return
        assert self.error is not None
        raise self.error


def always(error: Exception) -> object:
    """Build a dispatcher that always raises the given error."""

    async def fail(_player_id: str) -> None:
        raise error

    return fail


async def _run(mass: object, operation_id: str, behaviors: dict[str, object]) -> dict[str, object]:
    """Submit and wait for an operation with one scripted behavior per player."""
    for pid in behaviors:
        make_player(mass, pid)

    async def dispatcher(player_id: str) -> None:
        await behaviors[player_id](player_id)

    install_fake_command(mass, "players/cmd/pause", dispatcher)
    await mass.player_commands.submit(
        operation_id=operation_id,
        command="players/cmd/pause",
        player_ids=list(behaviors),
    )
    return await wait_for_terminal(mass, operation_id)


async def test_transient_unavailable_then_success_retries(pc_mass: object) -> None:
    """A player-unavailable response is retried with backoff until it succeeds."""
    mass = pc_mass
    flaky = AttemptBehavior(PlayerUnavailableError("nope"), succeed_on_attempt=3)
    row = await _run(mass, "retry-unavailable", {"flaky": flaky})
    assert row["status"] == "succeeded"
    assert flaky.calls == 3
    snap = target_map(await mass.player_commands.build_snapshot(row))
    assert snap["flaky"].attempts == 3
    events = await mass.player_commands.db.get_event_rows(
        after_cursor=0, operation_id="retry-unavailable", limit=100
    )
    assert any(event["event_type"] == "target_retrying" for event in events)


async def test_timeout_is_retried_then_succeeds(pc_mass: object) -> None:
    """A command timeout is treated as a transient failure and retried."""
    mass = pc_mass
    flaky = AttemptBehavior(TimeoutError("slow"), succeed_on_attempt=2)
    row = await _run(mass, "retry-timeout", {"flaky": flaky})
    assert row["status"] == "succeeded"
    assert flaky.calls == 2


async def test_rate_limited_and_busy_are_retried(pc_mass: object) -> None:
    """Rate limiting and resource-busy responses go through the retry budget."""
    mass = pc_mass
    limited = AttemptBehavior(RateLimited("slow down"), succeed_on_attempt=2)
    busy = AttemptBehavior(ResourceBusyError("busy"), succeed_on_attempt=2)
    row = await _run(mass, "retry-rate", {"limited": limited, "busy": busy})
    assert row["status"] == "succeeded"
    assert limited.calls == busy.calls == 2


async def test_unsupported_feature_is_terminal_on_first_attempt(pc_mass: object) -> None:
    """An unsupported-feature refusal terminates immediately without a retry."""
    mass = pc_mass
    row = await _run(
        mass,
        "terminal-unsupported",
        {"p": always(UnsupportedFeaturedException("no pause"))},
    )
    assert row["status"] == "failed"
    snap = target_map(await mass.player_commands.build_snapshot(row))
    assert snap["p"].attempts == 1
    assert snap["p"].last_error_code == 9


async def test_player_command_failed_is_terminal(pc_mass: object) -> None:
    """A deterministic device refusal (PlayerCommandFailed) is not retried."""
    mass = pc_mass
    row = await _run(
        mass, "terminal-refused", {"p": always(PlayerCommandFailed("device says no"))}
    )
    assert row["status"] == "failed"
    snap = target_map(await mass.player_commands.build_snapshot(row))
    assert snap["p"].attempts == 1
    assert snap["p"].last_error_code == 11


async def test_unknown_exception_consumes_retry_budget(pc_mass: object) -> None:
    """Unknown provider errors are retried until the attempt budget is exhausted."""
    mass = pc_mass
    mass.config.set_raw_core_config_value("player_commands", "max_attempts", 2)
    row = await _run(mass, "budget-unknown", {"p": always(RuntimeError("wat"))})
    assert row["status"] == "failed"
    snap = target_map(await mass.player_commands.build_snapshot(row))
    assert snap["p"].attempts == 2
    assert snap["p"].status == "failed"


async def test_retry_budget_is_isolated_per_target(pc_mass: object) -> None:
    """Exhausting one target's budget does not stop another target succeeding."""
    mass = pc_mass
    mass.config.set_raw_core_config_value("player_commands", "max_attempts", 2)
    row = await _run(
        mass,
        "budget-isolated",
        {"good": AttemptBehavior(None, 1), "bad": always(PlayerUnavailableError("x"))},
    )
    # the good target succeeds despite the bad one exhausting its own budget
    assert row["status"] == "partial"
    snap = target_map(await mass.player_commands.build_snapshot(row))
    assert snap["good"].status == "succeeded"
    assert snap["good"].attempts == 1
    assert snap["bad"].status == "failed"
    assert snap["bad"].attempts == 2
