"""
Fair dispatcher for asynchronous batch player commands.

One dispatch loop admits due targets under a global, a per-provider-instance and a
per-operation concurrency budget (FIFO within an operation, round-robin across
operations). Each admitted target gets a persisted lease and runs as its own task:
the player command is wrapped with a timeout, a purpose-scoped player lock and
error classification. Transient failures go to bounded exponential backoff,
deterministic refusals terminate immediately. Late callbacks converge through the
lease generation compare-and-swap in the database layer.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from typing import TYPE_CHECKING, Any, cast

from music_assistant_models.errors import MusicAssistantError

from music_assistant.controllers.player_commands.audit import log_audit
from music_assistant.controllers.player_commands.constants import (
    CONF_PC_COMMAND_TIMEOUT,
    CONF_PC_DEFAULT_OPERATION_CONCURRENCY,
    CONF_PC_LEASE_GRACE,
    CONF_PC_MAX_GLOBAL_CONCURRENCY,
    CONF_PC_MAX_PROVIDER_CONCURRENCY,
    CONF_PC_RETRY_BASE,
    CONF_PC_RETRY_CAP,
    DEFAULT_CONFIG,
    ERROR_PLAYER_UNAVAILABLE,
    ERROR_PROVIDER_INSTANCE_CHANGED,
    TargetStatus,
)
from music_assistant.controllers.player_commands.registry import COMMAND_REGISTRY
from music_assistant.controllers.webserver.helpers.auth_middleware import (
    current_user,
    has_player_access,
)
from music_assistant.helpers.api import parse_arguments
from music_assistant.helpers.json import json_loads

if TYPE_CHECKING:
    from collections.abc import Mapping

    from music_assistant.controllers.player_commands.controller import (
        PlayerCommandsController,
    )

# error codes that reject a command deterministically - retrying cannot help
_DETERMINISTIC_ERROR_CODES: frozenset[int] = frozenset({3, 9, 11, 12, 19, 22})
# error codes that describe a temporary condition worth a retry
_TRANSIENT_ERROR_CODES: frozenset[int] = frozenset({1, 10, 17, 24, 25})
_LOOP_IDLE_TIMEOUT = 1.0
_LOOP_BUSY_TIMEOUT = 0.05


class CommandScheduler:
    """Dispatch loop, concurrency accounting and target execution for batch commands."""

    def __init__(self, controller: PlayerCommandsController) -> None:
        """Bind the scheduler to its owning batch commands controller."""
        self._controller = controller
        self._mass = controller.mass
        self._logger = controller.logger
        self._wake = asyncio.Event()
        self._global_inflight = 0
        self._provider_inflight: dict[str, int] = {}
        self._op_inflight: dict[str, int] = {}
        self._rr_offset = 0
        self._dispatch_task: asyncio.Task[None] | None = None
        self._watchdog_task: asyncio.Task[None] | None = None
        self._exec_tasks: set[asyncio.Task[None]] = set()
        self._closing = False

    def wake(self) -> None:
        """Request an immediate dispatch pass."""
        self._wake.set()

    async def start(self) -> None:
        """Start the dispatch loop and lease watchdog."""
        self._closing = False
        self._dispatch_task = cast(
            "asyncio.Task[None]", self._mass.create_task(self._dispatch_loop())
        )
        self._watchdog_task = cast(
            "asyncio.Task[None]", self._mass.create_task(self._lease_watchdog())
        )

    async def stop(self) -> None:
        """Stop dispatching; in-flight commands are cancelled and recovered on reboot."""
        self._closing = True
        for exec_task in self._exec_tasks:
            exec_task.cancel()
        for lifecycle_task in (self._dispatch_task, self._watchdog_task):
            if lifecycle_task is not None:
                lifecycle_task.cancel()
        self._exec_tasks.clear()
        self._global_inflight = 0
        self._provider_inflight.clear()
        self._op_inflight.clear()

    @property
    def lease_owner(self) -> str:
        """Return the lease owner identity of this boot."""
        return self._controller.lease_owner

    # ------------------------------------------------------------ configuration

    def _conf(self, key: str) -> Any:
        default = DEFAULT_CONFIG[key]
        value = self._mass.config.get_raw_core_config_value("player_commands", key, default)
        try:
            return type(default)(value)
        except TypeError, ValueError:
            return default

    # ------------------------------------------------------------ dispatch loop

    async def _dispatch_loop(self) -> None:
        while not self._closing:
            self._wake.clear()
            try:
                await self._admit_pass()
                timeout = await self._next_wake_timeout()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._logger.exception("Batch command dispatch pass failed")
                timeout = _LOOP_IDLE_TIMEOUT
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(self._wake.wait(), timeout=timeout)

    async def _admit_pass(self) -> None:
        """Run round-robin admission rounds until every slot is taken or blocked."""
        while not self._closing:
            if self._global_inflight >= self._conf(CONF_PC_MAX_GLOBAL_CONCURRENCY):
                return
            operations = [
                op
                for op in await self._controller.db.get_active_operations()
                if op["status"] in ("queued", "running")
            ]
            if not operations:
                return
            operations.sort(key=lambda op: op["created_at"])
            offset = self._rr_offset % len(operations)
            operations = operations[offset:] + operations[:offset]
            progress = False
            for op in operations:
                if self._global_inflight >= self._conf(CONF_PC_MAX_GLOBAL_CONCURRENCY):
                    return
                operation_id = op["operation_id"]
                op_limit = op["max_concurrency"] or self._conf(
                    CONF_PC_DEFAULT_OPERATION_CONCURRENCY
                )
                if self._op_inflight.get(operation_id, 0) >= int(op_limit):
                    continue
                due = await self._controller.db.get_due_target(operation_id, time.time())
                if due is None:
                    continue
                player = self._mass.players.get_player(due["player_id"])
                if player is None or not player.initialized.is_set():
                    # unresolved players wait for a registry notification or the seal
                    # window; a registered-but-disabled player is admitted so its
                    # deterministic failure converges to a terminal state
                    continue
                provider_instance = player.provider.instance_id
                if self._provider_inflight.get(provider_instance, 0) >= self._conf(
                    CONF_PC_MAX_PROVIDER_CONCURRENCY
                ):
                    continue
                now = time.time()
                lease_expires_at = (
                    now + self._conf(CONF_PC_COMMAND_TIMEOUT) + self._conf(CONF_PC_LEASE_GRACE)
                )
                admitted = await self._controller.db.admit_target(
                    target=due,
                    provider_instance=provider_instance,
                    provider_domain=player.provider.domain,
                    lease_owner=self.lease_owner,
                    lease_expires_at=lease_expires_at,
                    now=now,
                )
                if admitted is None:
                    continue
                self._global_inflight += 1
                self._provider_inflight[provider_instance] = (
                    self._provider_inflight.get(provider_instance, 0) + 1
                )
                self._op_inflight[operation_id] = self._op_inflight.get(operation_id, 0) + 1
                task = self._mass.create_task(self._run_target(admitted))
                self._exec_tasks.add(task)
                task.add_done_callback(self._exec_tasks.discard)
                progress = True
            self._rr_offset += 1
            if not progress:
                return

    async def _next_wake_timeout(self) -> float:
        # retries are driven by their stored next_run_at, not by sleeps in tasks
        earliest = await self._controller.earliest_next_run_at()
        if earliest is None:
            return _LOOP_IDLE_TIMEOUT
        return min(_LOOP_IDLE_TIMEOUT, max(_LOOP_BUSY_TIMEOUT, earliest - time.time()))

    async def _lease_watchdog(self) -> None:
        while not self._closing:
            await asyncio.sleep(float(self._conf(CONF_PC_LEASE_GRACE)))
            try:
                requeued = await self._controller.db.expire_own_leases(
                    owner=self.lease_owner, now=time.time()
                )
                if requeued:
                    self._logger.warning(
                        "Requeued %d batch target(s) whose lease expired (stuck command?)",
                        requeued,
                    )
                    self.wake()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._logger.exception("Batch command lease watchdog failed")

    # ------------------------------------------------------------ target execution

    async def _run_target(self, target: Mapping[str, Any]) -> None:
        """Execute one leased target, converging it to its unique terminal state."""
        user_token: Any = None
        try:
            op = await self._controller.db.get_operation_row(target["operation_id"])
            if op is None:
                return
            user = None
            if op["created_by_user_id"]:
                user = await self._mass.webserver.auth.get_user(op["created_by_user_id"])
            # a background task does not inherit the submit request's ContextVars; set
            # the frozen owner explicitly so command-internal permission checks apply
            user_token = current_user.set(user)
            await self._execute_target(target, op)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.exception(
                "Unexpected failure running batch target %s", target["player_id"]
            )
        finally:
            if user_token is not None:
                current_user.reset(user_token)
            self._release_slot(target)
            self.wake()

    def _release_slot(self, target: Mapping[str, Any]) -> None:
        self._global_inflight = max(0, self._global_inflight - 1)
        operation_id = target["operation_id"]
        if operation_id in self._op_inflight:
            self._op_inflight[operation_id] = max(0, self._op_inflight[operation_id] - 1)
        instance = target["provider_instance"]
        if instance and instance in self._provider_inflight:
            self._provider_inflight[instance] = max(0, self._provider_inflight[instance] - 1)

    async def _execute_target(self, target: Mapping[str, Any], op: Mapping[str, Any]) -> None:
        player_id = target["player_id"]
        generation = int(target["lease_generation"])
        players = self._mass.players

        player = players.get_player(player_id)
        if player is None:
            if self._controller.is_permanently_gone(player_id):
                await self._controller.db.finish_target(
                    target_id=target["id"],
                    generation=generation,
                    status=TargetStatus.UNREACHABLE,
                    now=time.time(),
                    error_code=1001,
                    error_message="Player was permanently removed",
                )
            else:
                await self._fail_or_retry(
                    target,
                    generation,
                    code=ERROR_PLAYER_UNAVAILABLE,
                    message="Player is not registered",
                )
            return
        if not player.state.enabled:
            await self._controller.db.finish_target(
                target_id=target["id"],
                generation=generation,
                status=TargetStatus.FAILED,
                now=time.time(),
                error_code=ERROR_PLAYER_UNAVAILABLE,
                error_message="Player is disabled",
            )
            return
        if (
            target["provider_instance"]
            and player.provider.instance_id != target["provider_instance"]
        ):
            # the player id was re-registered behind a different provider identity
            await self._controller.db.finish_target(
                target_id=target["id"],
                generation=generation,
                status=TargetStatus.REPLACED,
                now=time.time(),
                error_code=ERROR_PROVIDER_INSTANCE_CHANGED,
                error_message="Player id is now backed by a different provider instance",
            )
            return
        user = current_user.get()
        if user is not None and not has_player_access(user, player_id, player):
            await self._controller.db.finish_target(
                target_id=target["id"],
                generation=generation,
                status=TargetStatus.FAILED,
                now=time.time(),
                error_code=22,
                error_message="No access to this player",
            )
            return

        spec = COMMAND_REGISTRY[str(op["command"])]
        stored_args = json_loads(op["args_json"]) if op["args_json"] else {}
        # re-run the real command's argument parsing so JSON scalars (enum values such
        # as repeat mode) are converted back to their typed forms for every attempt
        api_handler = self._mass.command_handlers[str(op["command"])]
        parsed_args = parse_arguments(
            api_handler.signature,
            api_handler.type_hints,
            {"player_id": player_id, **stored_args},
        )
        parsed_args.pop("player_id", None)
        # group playback owner redirect: record where the command really lands
        effective_player_id = player_id
        outcome = "direct"
        replaced_after_dispatch = False
        try:
            owner = players._get_player_with_redirect(player_id)
            if owner.player_id != player_id:
                effective_player_id = owner.player_id
                outcome = "redirected"
            handler = spec.bind(players)
            async with players.get_player_lock(player_id, spec.lock_purpose):
                async with asyncio.timeout(self._conf(CONF_PC_COMMAND_TIMEOUT)):
                    await handler(player_id, **parsed_args)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            await self._handle_command_error(target, generation, err)
            return

        # a re-registration during the in-flight command does not flip the result: the
        # device that received the command did receive it - annotate and report
        current = players.get_player(player_id)
        if current is not None and current is not player:
            replaced_after_dispatch = True
        applied = await self._controller.db.finish_target(
            target_id=target["id"],
            generation=generation,
            status=TargetStatus.SUCCEEDED,
            now=time.time(),
            outcome=outcome,
            effective_player_id=effective_player_id,
            event_payload_extra=(
                {"replaced_after_dispatch": True} if replaced_after_dispatch else None
            ),
        )
        if applied:
            log_audit(
                self._logger,
                "target_succeeded",
                operation_id=str(op["operation_id"]),
                actor=op["created_by_name"],
                player_id=player_id,
                attempt=int(target["attempts"]),
                outcome=outcome,
                effective=effective_player_id,
            )

    async def _handle_command_error(
        self,
        target: Mapping[str, Any],
        generation: int,
        err: Exception,
    ) -> None:
        # a permanent removal while the command was in flight: do not retry
        if self._controller.is_permanently_gone(str(target["player_id"])):
            await self._controller.db.finish_target(
                target_id=target["id"],
                generation=generation,
                status=TargetStatus.UNREACHABLE,
                now=time.time(),
                error_code=1001,
                error_message="Player was permanently removed",
            )
            return
        if isinstance(err, MusicAssistantError):
            code = int(err.error_code)
            message = str(err) or err.__class__.__name__
        elif isinstance(err, TimeoutError):
            code = ERROR_PLAYER_UNAVAILABLE
            message = f"Command timed out after {self._conf(CONF_PC_COMMAND_TIMEOUT)}s"
        else:
            # unknown provider errors are treated as transient until the budget runs out
            code = ERROR_PLAYER_UNAVAILABLE
            message = f"{type(err).__name__}: {err}"[:1000]
        await self._fail_or_retry(target, generation, code=code, message=message)

    async def _fail_or_retry(
        self,
        target: Mapping[str, Any],
        generation: int,
        *,
        code: int,
        message: str,
    ) -> None:
        op = await self._controller.db.get_operation_row(str(target["operation_id"]))
        if op is not None and int(op["cancel_requested"]) == 1:
            # the operation was cancelled while this command was in flight: no new
            # dispatch may happen, converge the target to its cancelled terminal state
            await self._controller.db.finish_target(
                target_id=target["id"],
                generation=generation,
                status=TargetStatus.CANCELLED,
                now=time.time(),
                error_code=code,
                error_message=message,
            )
            return
        attempts = int(target["attempts"])
        max_attempts = int(target["max_attempts"])
        if code in _DETERMINISTIC_ERROR_CODES or attempts >= max_attempts:
            # deterministic refusals terminate on the first attempt; transient and
            # unknown failures terminate once the attempt budget is exhausted
            await self._controller.db.finish_target(
                target_id=target["id"],
                generation=generation,
                status=TargetStatus.FAILED,
                now=time.time(),
                error_code=code,
                error_message=message,
            )
            return
        base = float(self._conf(CONF_PC_RETRY_BASE))
        cap = float(self._conf(CONF_PC_RETRY_CAP))
        delay = min(cap, base * (2 ** (attempts - 1))) * random.uniform(0.85, 1.15)
        now = time.time()
        applied = await self._controller.db.retry_target(
            target_id=target["id"],
            generation=generation,
            next_run_at=now + delay,
            error_code=code,
            error_message=message,
            now=now,
        )
        if applied:
            log_audit(
                self._logger,
                "target_retrying",
                operation_id=str(target["operation_id"]),
                actor="system",
                player_id=str(target["player_id"]),
                attempt=attempts,
                error_code=code,
                delay=round(delay, 3),
            )
