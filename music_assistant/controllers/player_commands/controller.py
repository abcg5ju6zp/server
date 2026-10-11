"""
Core controller for asynchronous batch player commands.

Exposes the client-facing batch protocol (submit / get / list / cancel / cursor
event subscription) and owns the scheduler, the dedicated player_commands.db
database and the startup reconciliation (lease takeover, sealing, retention).
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import orjson
from music_assistant_models.auth import Scope
from music_assistant_models.background_task import TaskSchedule
from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import ConfigEntryType
from music_assistant_models.errors import (
    InsufficientPermissions,
    InvalidCommand,
    InvalidDataError,
    PlayerUnavailableError,
)

from music_assistant.controllers.player_commands.audit import log_audit
from music_assistant.controllers.player_commands.constants import (
    CONF_PC_COMMAND_TIMEOUT,
    CONF_PC_DEFAULT_OPERATION_CONCURRENCY,
    CONF_PC_EVENT_GRACE_HOURS,
    CONF_PC_LEASE_GRACE,
    CONF_PC_MAX_ATTEMPTS,
    CONF_PC_MAX_GLOBAL_CONCURRENCY,
    CONF_PC_MAX_PROVIDER_CONCURRENCY,
    CONF_PC_PROVIDER_GRACE,
    CONF_PC_RETENTION_DAYS,
    CONF_PC_RETRY_BASE,
    CONF_PC_RETRY_CAP,
    DEFAULT_CONFIG,
    DEFAULT_EVENT_PAGE_LIMIT,
    DEFAULT_EVENT_WAIT_TIMEOUT,
    ERROR_INSUFFICIENT_PERMISSIONS,
    ERROR_PLAYER_UNAVAILABLE,
    MAX_EVENT_PAGE_LIMIT,
    MAX_OPERATION_ID_LENGTH,
    MAX_TARGETS_PER_OPERATION,
    TERMINAL_OPERATION_STATUSES,
    TargetStatus,
)
from music_assistant.controllers.player_commands.database import (
    PlayerCommandsDatabaseMixin,
)
from music_assistant.controllers.player_commands.errors import (
    BatchOperationConflictError,
)
from music_assistant.controllers.player_commands.models import (
    EventPage,
    OperationEvent,
    OperationRecord,
    OperationSnapshot,
    SubmitResult,
    TargetCounts,
    TargetRecord,
)
from music_assistant.controllers.player_commands.registry import COMMAND_REGISTRY
from music_assistant.controllers.player_commands.scheduler import CommandScheduler
from music_assistant.controllers.webserver.helpers.auth_middleware import (
    get_current_user,
    has_player_access,
    has_scope,
)
from music_assistant.helpers.api import api_command, parse_arguments
from music_assistant.helpers.json import json_loads
from music_assistant.models.core_controller import CoreController

_SEAL_TIMER_TASK_ID = "player_commands_seal_unresolved"
_RETENTION_TASK_ID = "player_commands_retention"
_OPERATION_ID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
_TERMINAL_STATUS_STRINGS = {status.value for status in TERMINAL_OPERATION_STATUSES}


class PlayerCommandsController(PlayerCommandsDatabaseMixin, CoreController):
    """Controller holding all logic for asynchronous batch player commands."""

    domain = "player_commands"

    def __init__(self, mass: Any) -> None:
        """Initialize controller state; the database is opened later in setup."""
        super().__init__(mass)
        self.manifest.name = "Batch Player Commands"
        self.manifest.description = (
            "Asynchronous batch control of multiple players with durable operation records"
        )
        self.manifest.icon = "select-multiple"
        self._database = None
        self._database_ready = False
        self._write_lock = asyncio.Lock()
        self._scheduler = CommandScheduler(self)
        self._boot_id = uuid.uuid4().hex
        self._events_wake = asyncio.Event()
        # player ids permanently removed (config gone) since this boot; cleared when an
        # equally named player registers again
        self._permanently_gone: set[str] = set()
        self._retention_registered = False

    # ------------------------------------------------ public interface

    @property
    def db(self) -> PlayerCommandsDatabaseMixin:
        """Return the database mixin (self) for callers that read as ``db``."""
        return self

    @property
    def lease_owner(self) -> str:
        """Return the lease owner identity (server id + boot id)."""
        return f"{self.mass.server_id}:{self._boot_id}"

    # ------------------------------------------------------------ lifecycle

    async def setup(self, config: Any) -> None:
        """Open the database and take over leases a previous process left behind."""
        await self.setup_database()
        if not self.database_ready:
            return
        await self.recover_on_startup(boot_owner=self.lease_owner, now=time.time())

    async def post_setup(self) -> None:
        """Start the dispatcher once all core controllers are available."""
        if not self.database_ready:
            return
        await self._scheduler.start()
        if not self._retention_registered:
            self.mass.tasks.register_scheduled_task(
                task_id=_RETENTION_TASK_ID,
                name="Cleanup expired batch player command operations",
                handler=self._cleanup_expired,
                schedule=TaskSchedule.daily(hour=4, minute=15),
                initial_delay=300,
            )
            self._retention_registered = True
        self.wake()

    async def on_providers_loaded(self) -> None:
        """
        Start the post-startup window after which never-resolved targets are sealed.

        Called by the server boot sequence once configured providers were loaded.
        """
        if not self.database_ready:
            return
        grace = self._conf(CONF_PC_PROVIDER_GRACE)
        if grace <= 0:
            self.mass.create_task(self.seal_unresolved())
        else:
            self.mass.call_later(float(grace), self.seal_unresolved, task_id=_SEAL_TIMER_TASK_ID)
        self.wake()

    async def close(self) -> None:
        """Stop dispatching and close the database on server exit."""
        self.mass.cancel_timer(_SEAL_TIMER_TASK_ID)
        await self._scheduler.stop()
        await self.close_database()

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return config entries tuning dispatch, retries and retention."""

        def entry(
            key: str,
            entry_type: ConfigEntryType,
            default: float,
            value_range: tuple[int, int] | None,
            *,
            advanced: bool = True,
        ) -> ConfigEntry:
            return ConfigEntry(
                key=key,
                type=entry_type,
                default_value=default,
                range=value_range,
                required=False,
                category="generic",
                advanced=advanced,
            )

        return (
            entry(
                CONF_PC_MAX_GLOBAL_CONCURRENCY,
                ConfigEntryType.INTEGER,
                16,
                (1, 100),
            ),
            entry(
                CONF_PC_MAX_PROVIDER_CONCURRENCY,
                ConfigEntryType.INTEGER,
                4,
                (1, 20),
            ),
            entry(
                CONF_PC_DEFAULT_OPERATION_CONCURRENCY,
                ConfigEntryType.INTEGER,
                8,
                (1, 50),
            ),
            entry(CONF_PC_COMMAND_TIMEOUT, ConfigEntryType.INTEGER, 30, (5, 300)),
            entry(CONF_PC_LEASE_GRACE, ConfigEntryType.INTEGER, 15, (5, 120)),
            entry(CONF_PC_MAX_ATTEMPTS, ConfigEntryType.INTEGER, 3, (1, 10)),
            entry(CONF_PC_RETRY_BASE, ConfigEntryType.FLOAT, 1.0, (0, 60)),
            entry(CONF_PC_RETRY_CAP, ConfigEntryType.INTEGER, 60, (0, 3600)),
            entry(CONF_PC_PROVIDER_GRACE, ConfigEntryType.INTEGER, 300, (0, 86400)),
            entry(CONF_PC_RETENTION_DAYS, ConfigEntryType.INTEGER, 7, (1, 90)),
            entry(CONF_PC_EVENT_GRACE_HOURS, ConfigEntryType.INTEGER, 24, (1, 168)),
        )

    # ------------------------------------------------------------ API commands

    @api_command(
        "player_commands/submit",
        required_scope=Scope.PLAYERS_CONTROL,
        allow_impersonation=True,
    )
    async def submit(
        self,
        operation_id: str,
        command: str,
        player_ids: list[str],
        args: dict[str, Any] | None = None,
        max_concurrency: int | None = None,
    ) -> SubmitResult:
        """
        Submit an asynchronous batch player command and get its operation record.

        A repeated submit with the same operation_id and an equivalent payload returns
        the original operation (reused=True); the same id with a different payload is a
        conflict. The call returns immediately with per-target states, never with a
        faked aggregate success.

        :param operation_id: Client chosen stable id ([A-Za-z0-9_-], max 64 chars).
        :param command: Single-player command path, e.g. players/cmd/pause.
        :param player_ids: Frozen target set (non empty, max 200, deduped).
        :param args: Keyword arguments for the command (without player_id).
        :param max_concurrency: Optional per-operation in-flight limit.
        """
        if not self.database_ready:
            msg = "Batch player commands database is unavailable"
            raise PlayerUnavailableError(msg)
        operation_id = self._validate_operation_id(operation_id)
        frozen_ids, parsed_args = self._validate_submit_request(
            command=command,
            player_ids=player_ids,
            raw_args=args or {},
            max_concurrency=max_concurrency,
        )
        args_hash = hashlib.sha256(
            orjson.dumps(parsed_args, option=orjson.OPT_SORT_KEYS)
        ).hexdigest()

        existing = await self.db.get_operation_row(operation_id)
        if existing is not None:
            mismatch = self._payload_mismatch(
                existing, command, args_hash, frozen_ids, max_concurrency
            )
            if mismatch:
                raise BatchOperationConflictError(operation_id, mismatch)
            snapshot = await self.build_snapshot(existing, user=get_current_user())
            return SubmitResult(snapshot=snapshot, reused=True)

        user = get_current_user()
        now = time.time()
        max_attempts = int(self._conf(CONF_PC_MAX_ATTEMPTS))
        targets = self._build_target_rows(frozen_ids, user, max_attempts)
        await self.db.create_operation(
            operation_id=operation_id,
            command=command,
            args=parsed_args,
            args_hash=args_hash,
            target_ids=frozen_ids,
            max_concurrency=max_concurrency,
            actor_user_id=user.user_id if user else None,
            actor_username=user.username if user else None,
            server_id=self.lease_owner,
            now=now,
            targets=targets,
        )
        log_audit(
            self.logger,
            "operation_created",
            operation_id=operation_id,
            actor=user.username if user else None,
            server_id=self.lease_owner,
            command=command,
            targets=len(frozen_ids),
        )
        row = await self.db.get_operation_row(operation_id)
        assert row is not None
        self.wake()
        return SubmitResult(snapshot=await self.build_snapshot(row))

    @api_command("player_commands/get", required_scope=Scope.PLAYERS_READ)
    async def get_operation(self, operation_id: str) -> OperationSnapshot:
        """
        Return the full snapshot of one batch operation.

        :param operation_id: The operation to fetch.
        """
        row = await self.db.get_operation_row(operation_id)
        if row is None:
            msg = f"Batch operation {operation_id} is not available"
            raise PlayerUnavailableError(msg)
        user = get_current_user()
        snapshot = await self.build_snapshot(row, user=user)
        if not await self._can_view_operation(row, snapshot.targets, user):
            msg = f"Batch operation {operation_id} is not available"
            raise PlayerUnavailableError(msg)
        return snapshot

    @api_command("player_commands/list", required_scope=Scope.PLAYERS_READ)
    async def list_operations(
        self,
        status: str | None = None,
        limit: int = 50,
        include_finished_within_hours: int | None = 24,
    ) -> list[OperationRecord]:
        """
        List batch operations visible to the caller (newest first).

        :param status: Optional status filter.
        :param limit: Maximum number of records.
        :param include_finished_within_hours: Include terminal operations this recent.
        """
        limit = max(1, min(int(limit), 200))
        user = get_current_user()
        result: list[OperationRecord] = []
        for row in await self.db.list_operation_rows(
            status=status,
            include_finished_within_hours=include_finished_within_hours,
            limit=limit * 3,
        ):
            target_rows = await self.db.get_target_rows(str(row["operation_id"]))
            visible_targets = [
                target
                for target in target_rows
                if self._target_visible(str(target["player_id"]), user)
            ]
            if await self._can_view_operation(row, visible_targets, user):
                result.append(self._to_record(row, target_rows))
            if len(result) >= limit:
                break
        return result

    @api_command(
        "player_commands/cancel",
        required_scope=Scope.PLAYERS_CONTROL,
        allow_impersonation=True,
    )
    async def cancel_operation(self, operation_id: str) -> OperationSnapshot:
        """
        Cancel a batch operation.

        Targets not yet dispatched reach a CANCELLED terminal state. Commands already
        sent to a device can not be withdrawn: their cancel boundary is recorded and
        their actual result converges to the normal terminal state.

        :param operation_id: The operation to cancel.
        """
        row = await self.db.get_operation_row(operation_id)
        if row is None:
            msg = f"Batch operation {operation_id} is not available"
            raise InvalidDataError(msg)
        user = get_current_user()
        targets = await self.db.get_target_rows(operation_id)
        if not await self._can_view_operation(row, targets, user):
            msg = f"{user.username if user else '?'} may not cancel operation {operation_id}"
            raise InsufficientPermissions(msg)
        if not await self._can_cancel_operation(row, targets, user):
            msg = f"{user.username if user else '?'} may not cancel operation {operation_id}"
            raise InsufficientPermissions(msg)
        updated = await self.db.request_cancel(
            operation_id=operation_id,
            actor_user_id=user.user_id if user else None,
            actor_username=user.username if user else None,
            server_id=self.lease_owner,
            now=time.time(),
        )
        log_audit(
            self.logger,
            "operation_cancelled",
            operation_id=operation_id,
            actor=user.username if user else None,
            server_id=self.lease_owner,
        )
        assert updated is not None
        self.wake()
        return await self.build_snapshot(updated, user=user)

    @api_command("player_commands/subscribe_events", required_scope=Scope.PLAYERS_READ)
    async def subscribe_events(
        self,
        operation_id: str | None = None,
        after_cursor: int | None = None,
        limit: int = DEFAULT_EVENT_PAGE_LIMIT,
        wait_timeout: float = DEFAULT_EVENT_WAIT_TIMEOUT,
    ) -> AsyncGenerator[EventPage]:
        """
        Subscribe to batch operation events from a cursor (long polling one page).

        Reconnect with the last received ``next_cursor`` to resume without losing or
        repeating events. When the cursor is older than the retained event window the
        response carries ``resync`` plus a current snapshot.

        :param operation_id: Optional operation filter (all visible ops when omitted).
        :param after_cursor: Last cursor acknowledged by the client.
        :param limit: Page size (max 1000).
        :param wait_timeout: Max seconds to wait for new events.
        """
        user = get_current_user()
        cursor = max(0, int(after_cursor or 0))
        limit = max(1, min(int(limit), MAX_EVENT_PAGE_LIMIT))
        wait_timeout = max(0.0, min(float(wait_timeout), 300.0))
        deadline = asyncio.get_running_loop().time() + wait_timeout
        while True:
            page = await self._build_event_page(user, operation_id, cursor, limit)
            if page is not None:
                yield page
                return
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                terminal = await self._is_terminal(operation_id)
                yield EventPage(events=[], next_cursor=cursor, terminal=terminal)
                return
            # clear + recheck closes the window in which an event could be missed
            self._events_wake.clear()
            page = await self._build_event_page(user, operation_id, cursor, limit)
            if page is not None:
                yield page
                return
            try:
                await asyncio.wait_for(self._events_wake.wait(), timeout=remaining)
            except TimeoutError:
                terminal = await self._is_terminal(operation_id)
                yield EventPage(events=[], next_cursor=cursor, terminal=terminal)
                return

    # ------------------------------------------------------------ topology hooks

    def notify_player_available(
        self, player_id: str, provider_instance: str, provider_domain: str
    ) -> None:
        """Report that a player registered (synchronous hook from the player controller)."""
        self._permanently_gone.discard(player_id)
        self.wake()

    def notify_player_gone(self, player_id: str, permanent: bool) -> None:
        """Report that a player unregistered (synchronous hook from the player controller)."""
        if permanent:
            self._permanently_gone.add(player_id)
            if self.database_ready:
                self.mass.create_task(self._settle_gone_targets(player_id))
        self.wake()

    def notify_player_replaced(
        self,
        player_id: str,
        old_provider_instance: str,
        old_provider_domain: str,
    ) -> None:
        """Report a same-id player object/provider replacement (synchronous hook)."""
        self._permanently_gone.discard(player_id)
        if self.database_ready:
            self.mass.create_task(self._settle_replaced_targets(player_id, old_provider_instance))
        self.wake()

    def is_permanently_gone(self, player_id: str) -> bool:
        """Return whether the player was permanently removed this boot."""
        return player_id in self._permanently_gone

    async def earliest_next_run_at(self) -> float | None:
        """Return the earliest pending retry time for the dispatch loop."""
        if not self.database_ready:
            return None
        return await self.db.get_earliest_next_run()

    def wake(self) -> None:
        """Wake both the dispatcher and event subscribers."""
        self._scheduler.wake()
        self._events_wake.set()

    # ------------------------------------------------------------ startup jobs

    async def seal_unresolved(self) -> None:
        """Seal pending targets that never resolved after the provider grace window."""
        if not self.database_ready:
            return
        unresolved = [
            target
            for target in await self.db.get_pending_unresolved_targets()
            if self.mass.players.get_player(str(target["player_id"])) is None
        ]
        if unresolved:
            await self.db.seal_targets(
                target_ids=[int(target["id"]) for target in unresolved], now=time.time()
            )
            self.logger.warning(
                "Sealed %d batch target(s) whose player never became available",
                len(unresolved),
            )
            self.wake()

    async def seal_now(self) -> None:
        """Run the unresolved-target sealing immediately (also used by tests)."""
        self.mass.cancel_timer(_SEAL_TIMER_TASK_ID)
        await self.seal_unresolved()

    async def build_snapshot(
        self, row: dict[str, Any] | Any, *, user: Any = None
    ) -> OperationSnapshot:
        """Build a snapshot model from an operation row, optionally target-filtered."""
        target_rows = await self.db.get_target_rows(str(row["operation_id"]))
        record = self._to_record(row, target_rows)
        # the creator always sees the full operation (including targets denied at
        # creation); other users only see targets they currently have access to
        if user is None or self._is_owner(row, user) or has_scope(user, Scope.ALL):
            visible_rows = target_rows
        else:
            visible_rows = [
                target
                for target in target_rows
                if self._target_visible(str(target["player_id"]), user)
            ]
        targets = [self._to_target(target) for target in visible_rows]
        next_cursor = await self.db.get_max_event_seq(str(row["operation_id"]))
        return OperationSnapshot(operation=record, targets=targets, next_cursor=next_cursor)

    # ------------------------------------------------ private helpers

    async def _cleanup_expired(self) -> None:
        if not self.database_ready:
            return
        retention_days = float(self._conf(CONF_PC_RETENTION_DAYS))
        event_grace_hours = float(self._conf(CONF_PC_EVENT_GRACE_HOURS))
        cutoff = time.time() - retention_days * 86400 - event_grace_hours * 3600
        await self.db.cleanup_terminal_operations(cutoff=cutoff)

    async def _settle_gone_targets(self, player_id: str) -> None:
        await self.db.settle_gone_targets(player_id=player_id, permanent=True, now=time.time())
        self.wake()

    async def _settle_replaced_targets(self, player_id: str, old_provider_instance: str) -> None:
        await self.db.settle_replaced_targets(
            player_id=player_id,
            old_provider_instance=old_provider_instance,
            now=time.time(),
        )
        self.wake()

    # ------------------------------------------------------------ snapshots/views

    def _to_record(self, row: Any, target_rows: list[Any]) -> OperationRecord:
        counts = TargetCounts()
        for target in target_rows:
            counts.total += 1
            status = str(target["status"])
            if status == TargetStatus.PENDING.value:
                counts.pending += 1
            elif status == TargetStatus.RUNNING.value:
                counts.running += 1
            elif status == TargetStatus.RETRY_WAIT.value:
                counts.retry_wait += 1
            elif status == TargetStatus.SUCCEEDED.value:
                counts.succeeded += 1
            elif status == TargetStatus.FAILED.value:
                counts.failed += 1
            elif status == TargetStatus.CANCELLED.value:
                counts.cancelled += 1
            elif status == TargetStatus.UNREACHABLE.value:
                counts.unreachable += 1
            elif status == TargetStatus.REPLACED.value:
                counts.replaced += 1
            elif status == TargetStatus.SEALED.value:
                counts.sealed += 1
        return OperationRecord(
            operation_id=str(row["operation_id"]),
            command=str(row["command"]),
            args=json_loads(row["args_json"]) if row["args_json"] else {},
            status=str(row["status"]),
            player_ids=json_loads(row["target_ids_json"]),
            max_concurrency=row["max_concurrency"],
            cancel_requested=bool(row["cancel_requested"]),
            created_by_user_id=row["created_by_user_id"],
            created_by_name=row["created_by_name"],
            created_at=float(row["created_at"]),
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            counts=counts,
        )

    def _to_target(self, row: Any) -> TargetRecord:
        return TargetRecord(
            player_id=str(row["player_id"]),
            status=str(row["status"]),
            outcome=row["outcome"],
            effective_player_id=row["effective_player_id"],
            provider_instance=str(row["provider_instance"] or ""),
            provider_domain=str(row["provider_domain"] or ""),
            attempts=int(row["attempts"]),
            max_attempts=int(row["max_attempts"]),
            next_run_at=row["next_run_at"],
            lease_expires_at=row["lease_expires_at"],
            last_error_code=row["last_error_code"],
            last_error_message=row["last_error_message"],
            cancel_boundary=bool(row["cancel_boundary"]),
            created_at=float(row["created_at"]),
            dispatched_at=row["dispatched_at"],
            finished_at=row["finished_at"],
        )

    def _target_visible(self, player_id: str, user: Any) -> bool:
        if user is None:
            return True
        return has_player_access(user, player_id, None)

    def _is_owner(self, row: Any, user: Any) -> bool:
        return bool(row["created_by_user_id"] and row["created_by_user_id"] == user.user_id)

    async def _can_view_operation(self, row: Any, visible_targets: list[Any], user: Any) -> bool:
        if user is None:
            return True
        if has_scope(user, Scope.ALL):
            return True
        if self._is_owner(row, user):
            return True
        return bool(visible_targets)

    async def _can_cancel_operation(self, row: Any, targets: list[Any], user: Any) -> bool:
        if user is None:
            return True
        if has_scope(user, Scope.ALL):
            return True
        if self._is_owner(row, user):
            return True
        # a user controlling at least one target may cancel the shared operation
        return any(self._target_visible(str(target["player_id"]), user) for target in targets)

    async def _is_terminal(self, operation_id: str | None) -> bool:
        if operation_id is None:
            return False
        row = await self.db.get_operation_row(operation_id)
        return row is not None and str(row["status"]) in _TERMINAL_STATUS_STRINGS

    # ------------------------------------------------------------ event pages

    async def _build_event_page(
        self, user: Any, operation_id: str | None, cursor: int, limit: int
    ) -> EventPage | None:
        if operation_id is not None:
            return await self._build_operation_event_page(user, operation_id, cursor, limit)
        return await self._build_global_event_page(user, cursor, limit)

    async def _build_operation_event_page(
        self, user: Any, operation_id: str, cursor: int, limit: int
    ) -> EventPage | None:
        row = await self.db.get_operation_row(operation_id)
        target_rows = [] if row is None else await self.db.get_target_rows(operation_id)
        if row is None or not await self._can_view_operation(
            row,
            [t for t in target_rows if self._target_visible(str(t["player_id"]), user)],
            user,
        ):
            return EventPage(
                events=[],
                next_cursor=cursor,
                resync=True,
                resync_reason="operation_gone",
                terminal=True,
            )
        min_seq = await self.db.get_min_event_seq(operation_id)
        if cursor > 0 and min_seq is not None and cursor < min_seq:
            snapshot = await self.build_snapshot(row, user=user)
            return EventPage(
                events=[],
                next_cursor=cursor,
                resync=True,
                resync_reason="gap",
                snapshot=snapshot,
            )
        event_rows = await self.db.get_event_rows(
            after_cursor=cursor, operation_id=operation_id, limit=limit
        )
        if not event_rows:
            if str(row["status"]) in _TERMINAL_STATUS_STRINGS:
                return EventPage(events=[], next_cursor=cursor, terminal=True)
            return None
        events = await self._filter_event_rows(event_rows, user, {operation_id: (row, target_rows)})
        last_seq = int(event_rows[-1]["seq"])
        max_seq = await self.db.get_max_event_seq(operation_id)
        terminal = max_seq <= last_seq and str(row["status"]) in _TERMINAL_STATUS_STRINGS
        return EventPage(events=events, next_cursor=last_seq, terminal=terminal)

    async def _build_global_event_page(
        self, user: Any, cursor: int, limit: int
    ) -> EventPage | None:
        min_seq = await self.db.get_min_event_seq(None)
        if cursor > 0 and min_seq is not None and cursor < min_seq:
            return EventPage(
                events=[],
                next_cursor=cursor,
                resync=True,
                resync_reason="gap",
            )
        event_rows = await self.db.get_event_rows(
            after_cursor=cursor, operation_id=None, limit=limit
        )
        if not event_rows:
            return None
        events = await self._filter_event_rows(event_rows, user, {})
        # the cursor advances over the full log position, including events hidden for
        # this user, so a filtered page does not trap the client in a replay loop
        return EventPage(events=events, next_cursor=int(event_rows[-1]["seq"]))

    async def _filter_event_rows(
        self,
        event_rows: list[Any],
        user: Any,
        op_cache: dict[str, tuple[Any, list[Any]]],
    ) -> list[OperationEvent]:
        """
        Filter persisted event rows for the current user.

        Target events pass the dynamic player filter (which may have changed since
        submit); operation-level events pass only when the operation itself is visible.
        """
        events: list[OperationEvent] = []
        for event_row in event_rows:
            operation_id = str(event_row["operation_id"])
            player_id = event_row["player_id"]
            if player_id is not None:
                if not self._target_visible(str(player_id), user):
                    continue
            elif user is not None:
                visible_op = op_cache.get(operation_id)
                if visible_op is None:
                    row = await self.db.get_operation_row(operation_id)
                    if row is None:
                        continue
                    target_rows = await self.db.get_target_rows(operation_id)
                    visible_op = (row, target_rows)
                    op_cache[operation_id] = visible_op
                row, target_rows = visible_op
                visible_targets = [
                    target
                    for target in target_rows
                    if self._target_visible(str(target["player_id"]), user)
                ]
                if not await self._can_view_operation(row, visible_targets, user):
                    continue
            events.append(
                OperationEvent(
                    cursor=int(event_row["seq"]),
                    operation_id=operation_id,
                    event_type=str(event_row["event_type"]),
                    created_at=float(event_row["created_at"]),
                    player_id=event_row["player_id"],
                    payload=json_loads(event_row["payload_json"])
                    if event_row["payload_json"]
                    else {},
                )
            )
        return events

    def _validate_submit_request(
        self,
        *,
        command: str,
        player_ids: list[str],
        raw_args: dict[str, Any],
        max_concurrency: int | None,
    ) -> tuple[list[str], dict[str, Any]]:
        """Validate command, target set and arguments; return (frozen ids, parsed args)."""
        if command not in COMMAND_REGISTRY:
            msg = f"Unsupported batch command: {command}"
            raise InvalidCommand(msg)
        handler = self.mass.command_handlers.get(command)
        if handler is None:
            msg = f"Unsupported batch command: {command}"
            raise InvalidCommand(msg)
        if not player_ids:
            msg = "player_ids must not be empty"
            raise InvalidDataError(msg)
        if len(player_ids) > MAX_TARGETS_PER_OPERATION:
            msg = f"A batch operation may hold at most {MAX_TARGETS_PER_OPERATION} targets"
            raise InvalidDataError(msg)
        if max_concurrency is not None and not 1 <= max_concurrency <= 50:
            msg = "max_concurrency must be between 1 and 50"
            raise InvalidDataError(msg)
        frozen_ids = sorted(set(player_ids))
        unknown = set(raw_args) - set(handler.signature.parameters)
        if unknown:
            msg = f"Unknown argument(s) for {command}: {', '.join(sorted(unknown))}"
            raise InvalidDataError(msg)
        try:
            parsed_args = parse_arguments(
                handler.signature,
                handler.type_hints,
                {"player_id": frozen_ids[0], **raw_args},
            )
        except (KeyError, TypeError, ValueError) as err:
            msg = f"Invalid arguments for {command}: {err}"
            raise InvalidDataError(msg) from err
        parsed_args.pop("player_id", None)
        if command == "players/cmd/group":
            # the group target sits outside the frozen set, so validate it here
            target_player = str(parsed_args.get("target_player") or "")
            group_target = self.mass.players.get_player(target_player)
            requester = get_current_user()
            if group_target is None or (
                requester is not None
                and not has_player_access(requester, target_player, group_target)
            ):
                msg = f"group target_player {target_player} is not available to the requester"
                raise InvalidDataError(msg)
        return frozen_ids, parsed_args

    # ------------------------------------------------------------ small helpers

    def _build_target_rows(
        self, frozen_ids: list[str], user: Any, max_attempts: int
    ) -> list[dict[str, Any]]:
        """Resolve each frozen target id to its initial persisted target state."""
        targets: list[dict[str, Any]] = []
        for player_id in frozen_ids:
            player = self.mass.players.get_player(player_id)
            target: dict[str, Any] = {
                "player_id": player_id,
                "status": TargetStatus.PENDING.value,
                "provider_instance": "",
                "provider_domain": "",
                "max_attempts": max_attempts,
            }
            if player is None:
                targets.append(target)
                continue
            target["provider_instance"] = player.provider.instance_id
            target["provider_domain"] = player.provider.domain
            if not player.state.enabled:
                target["status"] = TargetStatus.FAILED.value
                target["error_code"] = ERROR_PLAYER_UNAVAILABLE
                target["error_message"] = "Player is disabled"
            elif user is not None and not has_player_access(user, player_id, player):
                target["status"] = TargetStatus.FAILED.value
                target["error_code"] = ERROR_INSUFFICIENT_PERMISSIONS
                target["error_message"] = "No access to this player"
            targets.append(target)
        return targets

    def _validate_operation_id(self, operation_id: str) -> str:
        if not operation_id or not isinstance(operation_id, str):
            msg = "operation_id is required"
            raise InvalidDataError(msg)
        if len(operation_id) > MAX_OPERATION_ID_LENGTH:
            msg = f"operation_id may be at most {MAX_OPERATION_ID_LENGTH} characters"
            raise InvalidDataError(msg)
        if any(char not in _OPERATION_ID_CHARS for char in operation_id):
            msg = "operation_id may only contain letters, digits, '_' and '-'"
            raise InvalidDataError(msg)
        return operation_id

    def _payload_mismatch(
        self,
        existing: Any,
        command: str,
        args_hash: str,
        frozen_ids: list[str],
        max_concurrency: int | None,
    ) -> str | None:
        if str(existing["command"]) != command:
            return f"command {existing['command']} != {command}"
        if str(existing["args_hash"]) != args_hash:
            return "arguments differ"
        if json_loads(existing["target_ids_json"]) != frozen_ids:
            return "target set differs"
        if existing["max_concurrency"] != max_concurrency:
            return "max_concurrency differs"
        return None

    def _conf(self, key: str) -> int | float:
        default = DEFAULT_CONFIG[key]
        value = self.mass.config.get_raw_core_config_value("player_commands", key, default)
        try:
            return type(default)(value)
        except TypeError, ValueError:
            return default
