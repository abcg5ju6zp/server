"""
Database lifecycle and persistence layer for the batch player commands controller.

Owns ``player_commands.db`` on a dedicated WAL connection (operational data, kept out
of library.db so a failure here can never reset the music library). All state changes
append their event and audit rows in the same SQLite transaction as the state update
they describe, which is what keeps the query views and the event log monotonic.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sqlite3
import time
from typing import TYPE_CHECKING, Any, cast

from music_assistant.helpers.database import DatabaseConnection
from music_assistant.helpers.json import json_dumps

from .constants import (
    AUDIT_OPERATION_CANCELLED,
    AUDIT_OPERATION_CREATED,
    AUDIT_OPERATION_FINISHED,
    AUDIT_TARGET_DISPATCHED,
    PC_DB_FILENAME,
    PC_DB_SCHEMA_VERSION,
    PC_TABLE_AUDIT,
    PC_TABLE_EVENTS,
    PC_TABLE_OPERATIONS,
    PC_TABLE_SETTINGS,
    PC_TABLE_TARGETS,
    TERMINAL_OPERATION_STATUSES,
    TERMINAL_TARGET_EVENT,
    TERMINAL_TARGET_STATUSES,
    BatchEventType,
    OperationStatus,
    TargetStatus,
)
from .migrations import migrate_database

if TYPE_CHECKING:
    import logging
    from collections.abc import Mapping

    from music_assistant import MusicAssistant

_TERMINAL_TARGET_LIST = ", ".join(f"'{s.value}'" for s in TERMINAL_TARGET_STATUSES)
_TERMINAL_OP_LIST = ", ".join(f"'{s.value}'" for s in TERMINAL_OPERATION_STATUSES)
_ACTIVE_OP_LIST = ", ".join(
    f"'{s.value}'"
    for s in (OperationStatus.QUEUED, OperationStatus.RUNNING, OperationStatus.CANCELLING)
)


async def create_player_commands_tables(database: DatabaseConnection) -> int:
    """
    Create the batch commands tables if missing and return the stored schema version.

    :param database: Connection the batch commands database is reachable on.
    :returns: The stored schema version, 0 for a new file.
    """
    await database.execute(
        f"""CREATE TABLE IF NOT EXISTS {PC_TABLE_SETTINGS}(
                [key] TEXT PRIMARY KEY,
                [value] TEXT,
                [type] TEXT
            );"""
    )
    version_row = await database.get_row(PC_TABLE_SETTINGS, {"key": "version"})
    prev_version = 0
    if version_row is not None and version_row["value"] is not None:
        prev_version = int(version_row["value"])
    await database.execute(
        f"""CREATE TABLE IF NOT EXISTS {PC_TABLE_OPERATIONS}(
                [operation_id] TEXT PRIMARY KEY,
                [command] TEXT NOT NULL,
                [args_json] TEXT NOT NULL DEFAULT '{{}}',
                [args_hash] TEXT NOT NULL,
                [target_ids_json] TEXT NOT NULL,
                [max_concurrency] INTEGER,
                [status] TEXT NOT NULL,
                [cancel_requested] INTEGER NOT NULL DEFAULT 0,
                [num_targets] INTEGER NOT NULL,
                [created_by_user_id] TEXT,
                [created_by_name] TEXT,
                [created_server_id] TEXT NOT NULL,
                [created_at] REAL NOT NULL,
                [started_at] REAL,
                [finished_at] REAL
            );"""
    )
    await database.execute(
        f"""CREATE TABLE IF NOT EXISTS {PC_TABLE_TARGETS}(
                [id] INTEGER PRIMARY KEY AUTOINCREMENT,
                [operation_id] TEXT NOT NULL,
                [player_id] TEXT NOT NULL,
                [provider_instance] TEXT NOT NULL DEFAULT '',
                [provider_domain] TEXT NOT NULL DEFAULT '',
                [status] TEXT NOT NULL,
                [outcome] TEXT,
                [effective_player_id] TEXT,
                [attempts] INTEGER NOT NULL DEFAULT 0,
                [max_attempts] INTEGER NOT NULL,
                [next_run_at] REAL,
                [lease_owner] TEXT,
                [lease_generation] INTEGER NOT NULL DEFAULT 0,
                [lease_expires_at] REAL,
                [last_error_code] INTEGER,
                [last_error_message] TEXT,
                [last_error_at] REAL,
                [cancel_boundary] INTEGER NOT NULL DEFAULT 0,
                [created_at] REAL NOT NULL,
                [dispatched_at] REAL,
                [finished_at] REAL,
                UNIQUE(operation_id, player_id)
            );"""
    )
    await database.execute(
        "CREATE INDEX IF NOT EXISTS idx_batch_targets_dispatch "
        f"ON {PC_TABLE_TARGETS}(status, next_run_at)"
    )
    await database.execute(
        "CREATE INDEX IF NOT EXISTS idx_batch_targets_lease "
        f"ON {PC_TABLE_TARGETS}(status, lease_expires_at)"
    )
    await database.execute(
        f"""CREATE TABLE IF NOT EXISTS {PC_TABLE_EVENTS}(
                [seq] INTEGER PRIMARY KEY AUTOINCREMENT,
                [operation_id] TEXT NOT NULL,
                [target_id] INTEGER,
                [player_id] TEXT,
                [event_type] TEXT NOT NULL,
                [payload_json] TEXT,
                [created_at] REAL NOT NULL
            );"""
    )
    await database.execute(
        f"CREATE INDEX IF NOT EXISTS idx_batch_events_op ON {PC_TABLE_EVENTS}(operation_id, seq)"
    )
    await database.execute(
        f"""CREATE TABLE IF NOT EXISTS {PC_TABLE_AUDIT}(
                [id] INTEGER PRIMARY KEY AUTOINCREMENT,
                [created_at] REAL NOT NULL,
                [actor_user_id] TEXT,
                [actor_username] TEXT,
                [server_id] TEXT NOT NULL,
                [action] TEXT NOT NULL,
                [operation_id] TEXT NOT NULL,
                [player_id] TEXT,
                [attempt] INTEGER,
                [detail_json] TEXT
            );"""
    )
    await database.execute(
        f"CREATE INDEX IF NOT EXISTS idx_batch_audit_op ON {PC_TABLE_AUDIT}(operation_id)"
    )
    await database.commit()
    return prev_version


class PlayerCommandsDatabaseMixin:
    """Database lifecycle and persistence for the batch player commands controller."""

    if TYPE_CHECKING:
        mass: MusicAssistant
        logger: logging.Logger
        _database: DatabaseConnection | None
        _database_ready: bool
        _write_lock: asyncio.Lock

    @property
    def database_ready(self) -> bool:
        """Return whether the batch commands database is open and ready."""
        return self._database_ready

    @property
    def database(self) -> DatabaseConnection:
        """Return the batch commands database connection."""
        assert self._database is not None
        return self._database

    async def setup_database(self) -> None:
        """
        Open player_commands.db, create its tables and run pending migrations.

        An unusable file is moved aside and replaced by an empty one; any other failure
        disables the batch controller until a restart instead of taking the server down.
        """
        db_path = os.path.join(self.mass.storage_path, PC_DB_FILENAME)
        self._database_ready = False
        try:
            try:
                prev_version = await self._open_database(db_path)
            except sqlite3.DatabaseError as err:
                if getattr(err, "sqlite_errorcode", 0) & 0xFF not in (
                    sqlite3.SQLITE_CORRUPT,
                    sqlite3.SQLITE_NOTADB,
                ):
                    raise
                self.logger.error(
                    "Batch commands database %s is unusable (%s); moving it aside",
                    db_path,
                    err,
                )
                await self._quarantine_database(db_path)
                prev_version = await self._open_database(db_path)
            if prev_version > PC_DB_SCHEMA_VERSION:
                msg = (
                    f"{PC_DB_FILENAME} schema version {prev_version} is newer than this "
                    f"build supports ({PC_DB_SCHEMA_VERSION}); upgrade Music Assistant"
                )
                raise sqlite3.DatabaseError(msg)
            if prev_version != PC_DB_SCHEMA_VERSION:
                await migrate_database(
                    self.database, self.logger, prev_version, PC_DB_SCHEMA_VERSION
                )
                await self.database.insert_or_replace(
                    PC_TABLE_SETTINGS,
                    {"key": "version", "value": str(PC_DB_SCHEMA_VERSION), "type": "str"},
                )
                await self.database.commit()
        except (sqlite3.Error, OSError, ValueError) as err:
            self.logger.error(
                "Batch player commands unavailable: %s (%s). Restart to retry",
                db_path,
                err,
            )
            return
        self._database_ready = True

    async def close_database(self) -> None:
        """Close the batch commands database connection."""
        self._database_ready = False
        if self._database is None:
            return
        database, self._database = self._database, None
        with contextlib.suppress(sqlite3.Error):
            await database.close()

    # ------------------------------------------------------------- read queries

    async def get_operation_row(self, operation_id: str) -> Mapping[str, Any] | None:
        """Return one operation row by id (or None when it does not exist)."""
        return await self.database.get_row(PC_TABLE_OPERATIONS, {"operation_id": operation_id})

    async def get_target_rows(self, operation_id: str) -> list[Mapping[str, Any]]:
        """Return all target rows of an operation ordered by their insertion id."""
        return await self.database.get_rows_from_query(
            f"SELECT * FROM {PC_TABLE_TARGETS} WHERE operation_id = :operation_id ORDER BY id",
            {"operation_id": operation_id},
            limit=0,
        )

    async def get_target_counts(self, operation_id: str) -> dict[str, int]:
        """Return current per-status target counts for an operation."""
        rows = await self.database.get_rows_from_query(
            f"SELECT status, COUNT(*) AS cnt FROM {PC_TABLE_TARGETS} "
            "WHERE operation_id = :operation_id GROUP BY status",
            {"operation_id": operation_id},
            limit=0,
        )
        counts = {
            "total": 0,
            TargetStatus.PENDING.value: 0,
            TargetStatus.RUNNING.value: 0,
            TargetStatus.RETRY_WAIT.value: 0,
            TargetStatus.SUCCEEDED.value: 0,
            TargetStatus.FAILED.value: 0,
            TargetStatus.CANCELLED.value: 0,
            TargetStatus.UNREACHABLE.value: 0,
            TargetStatus.REPLACED.value: 0,
            TargetStatus.SEALED.value: 0,
        }
        for row in rows:
            counts[str(row["status"])] = int(row["cnt"])
            counts["total"] += int(row["cnt"])
        return counts

    async def get_active_operations(self) -> list[Mapping[str, Any]]:
        """Return queued/running/cancelling operations ordered by creation time."""
        return await self.database.get_rows_from_query(
            f"SELECT * FROM {PC_TABLE_OPERATIONS} WHERE status IN ({_ACTIVE_OP_LIST}) "
            "ORDER BY created_at, operation_id",
            limit=0,
        )

    async def get_earliest_next_run(self) -> float | None:
        """Return the earliest retry wake time across all operations, if any."""
        rows = await self.database.get_rows_from_query(
            f"SELECT MIN(next_run_at) AS min_next_run FROM {PC_TABLE_TARGETS} "
            "WHERE status = 'retry_wait' AND next_run_at IS NOT NULL",
            limit=1,
        )
        if not rows or rows[0]["min_next_run"] is None:
            return None
        return float(rows[0]["min_next_run"])

    async def get_pending_unresolved_targets(self) -> list[Mapping[str, Any]]:
        """Return never-dispatched pending targets (startup sealing candidates)."""
        return await self.database.get_rows_from_query(
            f"SELECT * FROM {PC_TABLE_TARGETS} WHERE status = 'pending' AND attempts = 0",
            limit=0,
        )

    async def get_due_target(self, operation_id: str, now: float) -> Mapping[str, Any] | None:
        """Return the next dispatchable (pending or due retry) target of an operation."""
        rows = await self.database.get_rows_from_query(
            f"SELECT * FROM {PC_TABLE_TARGETS} WHERE operation_id = :operation_id "
            "AND (status = 'pending' OR (status = 'retry_wait' AND next_run_at <= :now)) "
            "ORDER BY id LIMIT 1",
            {"operation_id": operation_id, "now": now},
            limit=0,
        )
        return rows[0] if rows else None

    async def list_operation_rows(
        self,
        *,
        status: str | None,
        include_finished_within_hours: float | None,
        limit: int,
    ) -> list[Mapping[str, Any]]:
        """List operation rows matching the given status/recency filters, newest first."""
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if status:
            clauses.append("status = :status")
            params["status"] = status
        if include_finished_within_hours is not None:
            cutoff = time.time() - include_finished_within_hours * 3600
            clauses.append(
                f"(status NOT IN ({_TERMINAL_OP_LIST}) OR finished_at >= :finished_cutoff)"
            )
            params["finished_cutoff"] = cutoff
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return await self.database.get_rows_from_query(
            f"SELECT * FROM {PC_TABLE_OPERATIONS}{where} ORDER BY created_at DESC LIMIT :limit",
            {**params, "limit": limit},
            limit=0,
        )

    # ------------------------------------------------------------- write queries

    async def create_operation(  # noqa: PLR0913
        self,
        *,
        operation_id: str,
        command: str,
        args: dict[str, Any],
        args_hash: str,
        target_ids: list[str],
        max_concurrency: int | None,
        actor_user_id: str | None,
        actor_username: str | None,
        server_id: str,
        now: float,
        targets: list[dict[str, Any]],
    ) -> None:
        """
        Persist a new operation and all its targets, events and audit rows atomically.

        Each target dict carries player_id, initial status, provider identity (empty
        strings when unresolved), max_attempts and optional normalized error fields.
        """
        async with self._write_tx() as db:
            await db.execute(
                f"INSERT INTO {PC_TABLE_OPERATIONS}"
                "(operation_id, command, args_json, args_hash, target_ids_json, max_concurrency, "
                "status, cancel_requested, num_targets, created_by_user_id, created_by_name, "
                "created_server_id, created_at) VALUES "
                "(:operation_id, :command, :args_json, :args_hash, :target_ids_json, "
                ":max_concurrency, :status, 0, :num_targets, :created_by_user_id, "
                ":created_by_name, :created_server_id, :created_at)",
                {
                    "operation_id": operation_id,
                    "command": command,
                    "args_json": json_dumps(args),
                    "args_hash": args_hash,
                    "target_ids_json": json_dumps(target_ids),
                    "max_concurrency": max_concurrency,
                    "status": OperationStatus.QUEUED.value,
                    "num_targets": len(targets),
                    "created_by_user_id": actor_user_id,
                    "created_by_name": actor_username,
                    "created_server_id": server_id,
                    "created_at": now,
                },
            )
            for target in targets:
                cursor = await db.execute(
                    f"INSERT INTO {PC_TABLE_TARGETS}"
                    "(operation_id, player_id, provider_instance, provider_domain, status, "
                    "attempts, max_attempts, last_error_code, last_error_message, created_at) "
                    "VALUES (:operation_id, :player_id, :provider_instance, :provider_domain, "
                    ":status, 0, :max_attempts, :last_error_code, :last_error_message, :created_at)",
                    {
                        "operation_id": operation_id,
                        "player_id": target["player_id"],
                        "provider_instance": target.get("provider_instance", ""),
                        "provider_domain": target.get("provider_domain", ""),
                        "status": target["status"],
                        "max_attempts": target["max_attempts"],
                        "last_error_code": target.get("error_code"),
                        "last_error_message": target.get("error_message"),
                        "created_at": now,
                    },
                )
                target_id = cursor.lastrowid
                if target["status"] != TargetStatus.PENDING.value:
                    # born-terminal targets (disabled / forbidden player) need an event
                    # so a subscriber from cursor zero observes their final state
                    event_type = TERMINAL_TARGET_EVENT[TargetStatus(target["status"])]
                    await self._insert_event(
                        db,
                        operation_id=operation_id,
                        target_id=target_id,
                        player_id=target["player_id"],
                        event_type=event_type,
                        created_at=now,
                        payload={
                            "attempts": 0,
                            "error_code": target.get("error_code"),
                            "error_message": target.get("error_message"),
                        },
                    )
            await self._insert_event(
                db,
                operation_id=operation_id,
                event_type=BatchEventType.OPERATION_CREATED,
                created_at=now,
                payload={
                    "command": command,
                    "player_ids": target_ids,
                    "max_concurrency": max_concurrency,
                },
            )
            await self._insert_audit(
                db,
                created_at=now,
                server_id=server_id,
                action=AUDIT_OPERATION_CREATED,
                operation_id=operation_id,
                actor_user_id=actor_user_id,
                actor_username=actor_username,
                detail={"command": command, "num_targets": len(targets)},
            )
            await self._maybe_finish_operation(db, operation_id, now)

    async def admit_target(
        self,
        *,
        target: Mapping[str, Any],
        provider_instance: str,
        provider_domain: str,
        lease_owner: str,
        lease_expires_at: float,
        now: float,
    ) -> Mapping[str, Any] | None:
        """
        Acure the lease for one due target (compare-and-swap).

        Returns the updated target row, or None when another path claimed it first
        (cancellation, watchdog expiry or a duplicate admission).
        """
        async with self._write_tx() as db:
            cursor = await db.execute(
                f"UPDATE {PC_TABLE_TARGETS} SET status = 'running', "
                "attempts = attempts + 1, dispatched_at = COALESCE(dispatched_at, :now), "
                "next_run_at = NULL, lease_owner = :lease_owner, "
                "lease_generation = lease_generation + 1, lease_expires_at = :lease_expires_at, "
                "provider_instance = :provider_instance, provider_domain = :provider_domain "
                "WHERE id = :target_id AND status IN ('pending', 'retry_wait') "
                "AND lease_owner IS NULL AND NOT EXISTS (SELECT 1 FROM "
                f"{PC_TABLE_OPERATIONS} o WHERE o.operation_id = {PC_TABLE_TARGETS}.operation_id "
                "AND o.cancel_requested = 1)",
                {
                    "now": now,
                    "lease_owner": lease_owner,
                    "lease_expires_at": lease_expires_at,
                    "provider_instance": provider_instance,
                    "provider_domain": provider_domain,
                    "target_id": target["id"],
                },
            )
            if cursor.rowcount == 0:
                return None
            row_cursor = await db.execute(
                f"SELECT * FROM {PC_TABLE_TARGETS} WHERE id = :target_id",
                {"target_id": target["id"]},
            )
            new_row = cast("Mapping[str, Any]", await row_cursor.fetchone())
            if new_row["attempts"] == 1:
                # first dispatch of the operation: mark it running
                await db.execute(
                    f"UPDATE {PC_TABLE_OPERATIONS} SET status = 'running', "
                    "started_at = COALESCE(started_at, :now) "
                    "WHERE operation_id = :operation_id AND status = 'queued'",
                    {"now": now, "operation_id": new_row["operation_id"]},
                )
                op_row = await self.database.get_row(
                    PC_TABLE_OPERATIONS, {"operation_id": new_row["operation_id"]}
                )
                if op_row is not None and op_row["status"] == OperationStatus.RUNNING.value:
                    await self._insert_event(
                        db,
                        operation_id=new_row["operation_id"],
                        event_type=BatchEventType.OPERATION_STATUS_CHANGED,
                        created_at=now,
                        payload={"status": OperationStatus.RUNNING.value},
                    )
            await self._insert_event(
                db,
                operation_id=new_row["operation_id"],
                target_id=new_row["id"],
                player_id=new_row["player_id"],
                event_type=BatchEventType.TARGET_DISPATCHED,
                created_at=now,
                payload={"attempt": new_row["attempts"], "lease_owner": lease_owner},
            )
            await self._insert_audit(
                db,
                created_at=now,
                server_id=lease_owner.split(":", 1)[0],
                action=AUDIT_TARGET_DISPATCHED,
                operation_id=new_row["operation_id"],
                actor_user_id=None,
                player_id=new_row["player_id"],
                attempt=int(new_row["attempts"]),
            )
            return new_row

    async def retry_target(
        self,
        *,
        target_id: int,
        generation: int,
        next_run_at: float,
        error_code: int,
        error_message: str,
        now: float,
    ) -> bool:
        """
        Move a leased target to RETRY_WAIT (compare-and-swap on the lease generation).

        :return: True when the retry was recorded, False when a stale callback lost the CAS.
        """
        async with self._write_tx() as db:
            cursor = await db.execute(
                f"UPDATE {PC_TABLE_TARGETS} SET status = 'retry_wait', next_run_at = :next_run_at, "
                "lease_owner = NULL, lease_expires_at = NULL, last_error_code = :error_code, "
                "last_error_message = :error_message, last_error_at = :now "
                "WHERE id = :target_id AND status = 'running' AND lease_generation = :generation",
                {
                    "next_run_at": next_run_at,
                    "error_code": error_code,
                    "error_message": error_message[:1000],
                    "now": now,
                    "target_id": target_id,
                    "generation": generation,
                },
            )
            if cursor.rowcount == 0:
                return False
            row = cast(
                "Mapping[str, Any]",
                await (
                    await db.execute(
                        f"SELECT * FROM {PC_TABLE_TARGETS} WHERE id = :target_id",
                        {"target_id": target_id},
                    )
                ).fetchone(),
            )
            await self._insert_event(
                db,
                operation_id=row["operation_id"],
                target_id=target_id,
                player_id=row["player_id"],
                event_type=BatchEventType.TARGET_RETRYING,
                created_at=now,
                payload={
                    "attempt": row["attempts"],
                    "next_run_at": next_run_at,
                    "delay": max(0.0, next_run_at - now),
                    "error_code": error_code,
                    "error_message": error_message[:500],
                },
            )
            return True

    async def finish_target(
        self,
        *,
        target_id: int,
        generation: int,
        status: TargetStatus,
        now: float,
        outcome: str | None = None,
        effective_player_id: str | None = None,
        error_code: int | None = None,
        error_message: str | None = None,
        event_payload_extra: dict[str, Any] | None = None,
    ) -> bool:
        """
        Converge a leased target to a terminal status (compare-and-swap on generation).

        :return: True when this call produced the terminal state, False for a stale or
            duplicate callback (the unique terminal state is then left untouched).
        """
        async with self._write_tx() as db:
            cursor = await db.execute(
                f"UPDATE {PC_TABLE_TARGETS} SET status = :status, outcome = :outcome, "
                "effective_player_id = :effective_player_id, lease_owner = NULL, "
                "lease_expires_at = NULL, finished_at = :now, last_error_code = :error_code, "
                "last_error_message = :error_message, last_error_at = :error_at "
                "WHERE id = :target_id AND status = 'running' AND lease_generation = :generation",
                {
                    "status": status.value,
                    "outcome": outcome,
                    "effective_player_id": effective_player_id,
                    "now": now,
                    "error_code": error_code,
                    "error_message": error_message[:1000] if error_message else None,
                    "error_at": now if error_code is not None else None,
                    "target_id": target_id,
                    "generation": generation,
                },
            )
            if cursor.rowcount == 0:
                return False
            row = cast(
                "Mapping[str, Any]",
                await (
                    await db.execute(
                        f"SELECT * FROM {PC_TABLE_TARGETS} WHERE id = :target_id",
                        {"target_id": target_id},
                    )
                ).fetchone(),
            )
            payload: dict[str, Any] = {
                "attempts": row["attempts"],
                "error_code": error_code,
                "error_message": error_message[:500] if error_message else None,
            }
            if outcome is not None:
                payload["outcome"] = outcome
                payload["effective_player_id"] = effective_player_id
            if event_payload_extra:
                payload.update(event_payload_extra)
            await self._insert_event(
                db,
                operation_id=row["operation_id"],
                target_id=target_id,
                player_id=row["player_id"],
                event_type=TERMINAL_TARGET_EVENT[status],
                created_at=now,
                payload=payload,
            )
            await self._maybe_finish_operation(db, row["operation_id"], now)
            return True

    async def request_cancel(
        self,
        *,
        operation_id: str,
        actor_user_id: str | None,
        actor_username: str | None,
        server_id: str,
        now: float,
    ) -> Mapping[str, Any] | None:
        """
        Record a cancellation request and converge all undispatched targets.

        Already-running targets only get the cancel_boundary marker: the device command
        can not be withdrawn, so its result is awaited and converges normally. Returns
        None when the operation does not exist; terminal operations are returned as-is
        (idempotent cancel).
        """
        async with self._write_tx() as db:
            op_row = cast(
                "Mapping[str, Any] | None",
                await (
                    await db.execute(
                        f"SELECT * FROM {PC_TABLE_OPERATIONS} WHERE operation_id = :operation_id",
                        {"operation_id": operation_id},
                    )
                ).fetchone(),
            )
            if op_row is None:
                return None
            if op_row["status"] in TERMINAL_OPERATION_STATUSES:
                return op_row
            await db.execute(
                f"UPDATE {PC_TABLE_OPERATIONS} SET cancel_requested = 1, status = 'cancelling' "
                "WHERE operation_id = :operation_id",
                {"operation_id": operation_id},
            )
            # cancel every target that was never dispatched to a device
            pending_rows = await db.execute(
                f"SELECT * FROM {PC_TABLE_TARGETS} WHERE operation_id = :operation_id "
                "AND status IN ('pending', 'retry_wait')",
                {"operation_id": operation_id},
            )
            for row in await pending_rows.fetchall():
                await db.execute(
                    f"UPDATE {PC_TABLE_TARGETS} SET status = 'cancelled', next_run_at = NULL, "
                    "lease_owner = NULL, lease_expires_at = NULL, finished_at = :now "
                    "WHERE id = :target_id",
                    {"now": now, "target_id": row["id"]},
                )
                await self._insert_event(
                    db,
                    operation_id=operation_id,
                    target_id=row["id"],
                    player_id=row["player_id"],
                    event_type=BatchEventType.TARGET_CANCELLED,
                    created_at=now,
                    payload={"attempts": row["attempts"], "before_dispatch": True},
                )
            # mark the non-withdrawable boundary for commands already at a device
            running_rows = await db.execute(
                f"SELECT * FROM {PC_TABLE_TARGETS} WHERE operation_id = :operation_id "
                "AND status = 'running'",
                {"operation_id": operation_id},
            )
            for row in await running_rows.fetchall():
                await db.execute(
                    f"UPDATE {PC_TABLE_TARGETS} SET cancel_boundary = 1 WHERE id = :target_id",
                    {"target_id": row["id"]},
                )
                await self._insert_event(
                    db,
                    operation_id=operation_id,
                    target_id=row["id"],
                    player_id=row["player_id"],
                    event_type=BatchEventType.CANCEL_BOUNDARY,
                    created_at=now,
                    payload={"attempt": row["attempts"], "generation": row["lease_generation"]},
                )
            await self._insert_event(
                db,
                operation_id=operation_id,
                event_type=BatchEventType.OPERATION_STATUS_CHANGED,
                created_at=now,
                payload={"status": OperationStatus.CANCELLING.value},
            )
            await self._insert_audit(
                db,
                created_at=now,
                server_id=server_id,
                action=AUDIT_OPERATION_CANCELLED,
                operation_id=operation_id,
                actor_user_id=actor_user_id,
                actor_username=actor_username,
            )
            await self._maybe_finish_operation(db, operation_id, now)
            return cast(
                "Mapping[str, Any] | None",
                await (
                    await db.execute(
                        f"SELECT * FROM {PC_TABLE_OPERATIONS} WHERE operation_id = :operation_id",
                        {"operation_id": operation_id},
                    )
                ).fetchone(),
            )

    async def recover_on_startup(self, *, boot_owner: str, now: float) -> None:
        """
        Take over leases a previous process left behind.

        All RUNNING targets were owned by the dead boot and go back to RETRY_WAIT with
        their attempt count preserved; terminal rows - notably SUCCEEDED - are never
        touched, so a confirmed command is never executed twice.
        """
        async with self._write_tx() as db:
            running = await db.execute(
                f"SELECT t.* FROM {PC_TABLE_TARGETS} t JOIN {PC_TABLE_OPERATIONS} o "
                "ON o.operation_id = t.operation_id WHERE t.status = 'running'"
            )
            for row in await running.fetchall():
                op_cursor = await db.execute(
                    f"SELECT cancel_requested FROM {PC_TABLE_OPERATIONS} "
                    "WHERE operation_id = :operation_id",
                    {"operation_id": row["operation_id"]},
                )
                op_row = await op_cursor.fetchone()
                if op_row is not None and int(op_row["cancel_requested"]) == 1:
                    # cancellation was requested before the crash and the device
                    # response is gone: it can not be withdrawn nor confirmed, cancel it
                    await db.execute(
                        f"UPDATE {PC_TABLE_TARGETS} SET status = 'cancelled', next_run_at = NULL, "
                        "lease_owner = NULL, lease_expires_at = NULL, finished_at = :now "
                        "WHERE id = :target_id",
                        {"now": now, "target_id": row["id"]},
                    )
                    await self._insert_event(
                        db,
                        operation_id=row["operation_id"],
                        target_id=row["id"],
                        player_id=row["player_id"],
                        event_type=BatchEventType.TARGET_CANCELLED,
                        created_at=now,
                        payload={
                            "attempts": row["attempts"],
                            "reason": "boot_recovery",
                        },
                    )
                    continue
                await db.execute(
                    f"UPDATE {PC_TABLE_TARGETS} SET status = 'retry_wait', next_run_at = :now, "
                    "lease_owner = NULL, lease_expires_at = NULL, "
                    "lease_generation = lease_generation + 1 WHERE id = :target_id",
                    {"now": now, "target_id": row["id"]},
                )
                await self._insert_event(
                    db,
                    operation_id=row["operation_id"],
                    target_id=row["id"],
                    player_id=row["player_id"],
                    event_type=BatchEventType.TARGET_RETRYING,
                    created_at=now,
                    payload={
                        "attempt": row["attempts"],
                        "reason": "boot_recovery",
                        "lease_owner": row["lease_owner"],
                    },
                )
            active = await db.execute(
                f"SELECT * FROM {PC_TABLE_OPERATIONS} WHERE status IN ({_ACTIVE_OP_LIST})"
            )
            for op in await active.fetchall():
                target_rows = await db.execute(
                    f"SELECT COUNT(*) AS dispatched FROM {PC_TABLE_TARGETS} "
                    "WHERE operation_id = :operation_id AND attempts > 0",
                    {"operation_id": op["operation_id"]},
                )
                dispatched_row = await target_rows.fetchone()
                new_status = (
                    OperationStatus.RUNNING.value
                    if dispatched_row is not None and int(dispatched_row["dispatched"]) > 0
                    else OperationStatus.QUEUED.value
                )
                if op["cancel_requested"]:
                    new_status = OperationStatus.CANCELLING.value
                await db.execute(
                    f"UPDATE {PC_TABLE_OPERATIONS} SET status = :status WHERE operation_id = :id",
                    {"status": new_status, "id": op["operation_id"]},
                )
                await self._maybe_finish_operation(db, op["operation_id"], now)
        self.logger.info("Recovered batch player command leases owned by previous boots")

    async def expire_own_leases(self, *, owner: str, now: float) -> int:
        """
        Requeue leased targets of this boot whose lease expired (watchdog safety net).

        The generation is bumped so a callback of the stuck attempt loses its CAS.
        """
        requeued = 0
        async with self._write_tx() as db:
            rows = await db.execute(
                f"SELECT t.* FROM {PC_TABLE_TARGETS} t JOIN {PC_TABLE_OPERATIONS} o "
                "ON o.operation_id = t.operation_id WHERE t.status = 'running' "
                "AND t.lease_owner = :owner AND t.lease_expires_at < :now",
                {"owner": owner, "now": now},
            )
            affected_operations: set[str] = set()
            for row in await rows.fetchall():
                op_cursor = await db.execute(
                    f"SELECT cancel_requested FROM {PC_TABLE_OPERATIONS} "
                    "WHERE operation_id = :operation_id",
                    {"operation_id": row["operation_id"]},
                )
                op_row = await op_cursor.fetchone()
                if op_row is not None and int(op_row["cancel_requested"]) == 1:
                    await db.execute(
                        f"UPDATE {PC_TABLE_TARGETS} SET status = 'cancelled', "
                        "lease_owner = NULL, lease_expires_at = NULL, finished_at = :now "
                        "WHERE id = :target_id",
                        {"now": now, "target_id": row["id"]},
                    )
                    await self._insert_event(
                        db,
                        operation_id=row["operation_id"],
                        target_id=row["id"],
                        player_id=row["player_id"],
                        event_type=BatchEventType.TARGET_CANCELLED,
                        created_at=now,
                        payload={"attempt": row["attempts"], "reason": "lease_expired_cancelled"},
                    )
                else:
                    await db.execute(
                        f"UPDATE {PC_TABLE_TARGETS} SET status = 'retry_wait', next_run_at = :now, "
                        "lease_owner = NULL, lease_expires_at = NULL, "
                        "lease_generation = lease_generation + 1 WHERE id = :target_id",
                        {"now": now, "target_id": row["id"]},
                    )
                    await self._insert_event(
                        db,
                        operation_id=row["operation_id"],
                        target_id=row["id"],
                        player_id=row["player_id"],
                        event_type=BatchEventType.TARGET_RETRYING,
                        created_at=now,
                        payload={"attempt": row["attempts"], "reason": "lease_expired"},
                    )
                    requeued += 1
                affected_operations.add(row["operation_id"])
            for operation_id in affected_operations:
                await self._maybe_finish_operation(db, operation_id, now)
        return requeued

    async def get_running_targets(self, owner: str) -> list[Mapping[str, Any]]:
        """Return targets currently leased by the given owner."""
        return await self.database.get_rows_from_query(
            f"SELECT * FROM {PC_TABLE_TARGETS} WHERE status = 'running' AND lease_owner = :owner",
            {"owner": owner},
            limit=0,
        )

    async def seal_targets(self, *, target_ids: list[int], now: float) -> None:
        """Seal never-resolved targets (no dispatch ever happened) as SEALED."""
        if not target_ids:
            return
        async with self._write_tx() as db:
            placeholders = ", ".join(f":id_{i}" for i in range(len(target_ids)))
            params = {f"id_{i}": target_id for i, target_id in enumerate(target_ids)}
            rows = await db.execute(
                f"SELECT * FROM {PC_TABLE_TARGETS} WHERE id IN ({placeholders}) "
                "AND status = 'pending' AND attempts = 0",
                params,
            )
            affected_operations: set[str] = set()
            for row in await rows.fetchall():
                await db.execute(
                    f"UPDATE {PC_TABLE_TARGETS} SET status = 'sealed', finished_at = :now "
                    "WHERE id = :target_id",
                    {"now": now, "target_id": row["id"]},
                )
                await self._insert_event(
                    db,
                    operation_id=row["operation_id"],
                    target_id=row["id"],
                    player_id=row["player_id"],
                    event_type=BatchEventType.TARGET_SEALED,
                    created_at=now,
                    payload={"attempts": 0},
                )
                affected_operations.add(row["operation_id"])
            for operation_id in affected_operations:
                await self._maybe_finish_operation(db, operation_id, now)

    async def settle_gone_targets(self, *, player_id: str, permanent: bool, now: float) -> None:
        """
        React to a player leaving the registry.

        Permanent removal terminates every queued target for the player as unreachable.
        A temporary unavailability leaves queued targets to retry.
        """
        if not permanent:
            return
        async with self._write_tx() as db:
            rows = await db.execute(
                f"SELECT * FROM {PC_TABLE_TARGETS} WHERE player_id = :player_id "
                "AND status IN ('pending', 'retry_wait')",
                {"player_id": player_id},
            )
            affected: list[Mapping[str, Any]] = list(await rows.fetchall())
            for row in affected:
                await db.execute(
                    f"UPDATE {PC_TABLE_TARGETS} SET status = 'unreachable', next_run_at = NULL, "
                    "lease_owner = NULL, lease_expires_at = NULL, finished_at = :now, "
                    "last_error_code = :code, last_error_message = :message, last_error_at = :now "
                    "WHERE id = :target_id",
                    {
                        "now": now,
                        "code": 1001,
                        "message": "Player was permanently removed",
                        "target_id": row["id"],
                    },
                )
                await self._insert_event(
                    db,
                    operation_id=row["operation_id"],
                    target_id=row["id"],
                    player_id=player_id,
                    event_type=BatchEventType.TARGET_UNREACHABLE,
                    created_at=now,
                    payload={"attempts": row["attempts"], "reason": "player_removed"},
                )
            for row in affected:
                await self._maybe_finish_operation(db, row["operation_id"], now)

    async def settle_replaced_targets(
        self, *, player_id: str, old_provider_instance: str, now: float
    ) -> None:
        """Terminate queued targets whose frozen provider identity was replaced."""
        async with self._write_tx() as db:
            rows = await db.execute(
                f"SELECT * FROM {PC_TABLE_TARGETS} WHERE player_id = :player_id "
                "AND status IN ('pending', 'retry_wait') AND provider_instance = :old_instance "
                "AND provider_instance != ''",
                {"player_id": player_id, "old_instance": old_provider_instance},
            )
            affected: list[Mapping[str, Any]] = list(await rows.fetchall())
            for row in affected:
                await db.execute(
                    f"UPDATE {PC_TABLE_TARGETS} SET status = 'replaced', next_run_at = NULL, "
                    "lease_owner = NULL, lease_expires_at = NULL, finished_at = :now, "
                    "last_error_code = :code, last_error_message = :message, last_error_at = :now "
                    "WHERE id = :target_id",
                    {
                        "now": now,
                        "code": 1002,
                        "message": "Player id is now backed by a different provider instance",
                        "target_id": row["id"],
                    },
                )
                await self._insert_event(
                    db,
                    operation_id=row["operation_id"],
                    target_id=row["id"],
                    player_id=player_id,
                    event_type=BatchEventType.TARGET_REPLACED,
                    created_at=now,
                    payload={"attempts": row["attempts"], "old_instance": old_provider_instance},
                )
            for row in affected:
                await self._maybe_finish_operation(db, row["operation_id"], now)

    # ----------------------------------------------------------------- event log

    async def get_min_event_seq(self, operation_id: str | None = None) -> int | None:
        """Return the lowest retained event seq (for one op or globally), if any."""
        if operation_id:
            query = (
                f"SELECT MIN(seq) AS min_seq FROM {PC_TABLE_EVENTS} "
                "WHERE operation_id = :operation_id"
            )
            params: dict[str, Any] = {"operation_id": operation_id}
        else:
            query = f"SELECT MIN(seq) AS min_seq FROM {PC_TABLE_EVENTS}"
            params = {}
        row = await self.database.get_rows_from_query(query, params, limit=1)
        if not row or row[0]["min_seq"] is None:
            return None
        return int(row[0]["min_seq"])

    async def get_max_event_seq(self, operation_id: str | None = None) -> int:
        """Return the highest event seq (for one op or globally), 0 when empty."""
        if operation_id:
            query = (
                f"SELECT MAX(seq) AS max_seq FROM {PC_TABLE_EVENTS} "
                "WHERE operation_id = :operation_id"
            )
            params: dict[str, Any] = {"operation_id": operation_id}
        else:
            query = f"SELECT MAX(seq) AS max_seq FROM {PC_TABLE_EVENTS}"
            params = {}
        row = await self.database.get_rows_from_query(query, params, limit=1)
        if not row or row[0]["max_seq"] is None:
            return 0
        return int(row[0]["max_seq"])

    async def get_event_rows(
        self, *, after_cursor: int, operation_id: str | None, limit: int
    ) -> list[Mapping[str, Any]]:
        """Return event rows strictly after the cursor in seq order."""
        clauses = ["seq > :cursor"]
        params: dict[str, Any] = {"cursor": after_cursor, "limit": limit}
        if operation_id:
            clauses.append("operation_id = :operation_id")
            params["operation_id"] = operation_id
        return await self.database.get_rows_from_query(
            f"SELECT * FROM {PC_TABLE_EVENTS} WHERE {' AND '.join(clauses)} "
            "ORDER BY seq LIMIT :limit",
            params,
            limit=0,
        )

    async def cleanup_terminal_operations(self, *, cutoff: float) -> int:
        """
        Delete fully terminal operations older than the safety cutoff, cascading rows.

        Only operations whose every target is terminal and whose finish time is beyond
        both the retention and the event grace window are removed, so a reconnecting
        subscriber keeps its event window available.
        """
        async with self._write_tx() as db:
            rows = await db.execute(
                f"SELECT operation_id FROM {PC_TABLE_OPERATIONS} "
                f"WHERE status IN ({_TERMINAL_OP_LIST}) AND finished_at IS NOT NULL "
                "AND finished_at < :cutoff AND NOT EXISTS (SELECT 1 FROM "
                f"{PC_TABLE_TARGETS} t WHERE t.operation_id = "
                f"{PC_TABLE_OPERATIONS}.operation_id AND t.status NOT IN ({_TERMINAL_TARGET_LIST}))",
                {"cutoff": cutoff},
            )
            operation_ids = [row[0] for row in await rows.fetchall()]
            for operation_id in operation_ids:
                for table in (PC_TABLE_TARGETS, PC_TABLE_EVENTS, PC_TABLE_AUDIT):
                    await db.execute(
                        f"DELETE FROM {table} WHERE operation_id = :operation_id",
                        {"operation_id": operation_id},
                    )
                await db.execute(
                    f"DELETE FROM {PC_TABLE_OPERATIONS} WHERE operation_id = :operation_id",
                    {"operation_id": operation_id},
                )
        if operation_ids:
            self.logger.info(
                "Cleaned up %d expired batch player command operations", len(operation_ids)
            )
        return len(operation_ids)

    # ------------------------------------------------- private primitives

    @contextlib.asynccontextmanager
    async def _write_tx(self) -> Any:
        """
        Serialize multi-statement writes and run them in one explicit transaction.

        Only raw ``execute()`` calls may be used inside: the insert/update helpers
        auto-commit and would break the atomicity this scope exists for.
        """
        async with self._write_lock:
            db = self.database
            await db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                await db.commit()
            except BaseException:
                with contextlib.suppress(sqlite3.Error):
                    await db._db.rollback()
                raise
            # give other tasks a chance between transactions so a hot writer can not
            # monopolize the single connection's worker thread
            await asyncio.sleep(0)

    async def _insert_event(
        self,
        db: DatabaseConnection,
        *,
        operation_id: str,
        event_type: BatchEventType,
        created_at: float,
        target_id: int | None = None,
        player_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> int:
        cursor = await db.execute(
            f"INSERT INTO {PC_TABLE_EVENTS}"
            "(operation_id, target_id, player_id, event_type, payload_json, created_at) "
            "VALUES (:operation_id, :target_id, :player_id, :event_type, :payload_json, :created_at)",
            {
                "operation_id": operation_id,
                "target_id": target_id,
                "player_id": player_id,
                "event_type": event_type.value,
                "payload_json": json_dumps(payload or {}),
                "created_at": created_at,
            },
        )
        assert cursor.lastrowid is not None
        return int(cursor.lastrowid)

    async def _insert_audit(
        self,
        db: DatabaseConnection,
        *,
        created_at: float,
        server_id: str,
        action: str,
        operation_id: str,
        actor_user_id: str | None = None,
        actor_username: str | None = None,
        player_id: str | None = None,
        attempt: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        await db.execute(
            f"INSERT INTO {PC_TABLE_AUDIT}"
            "(created_at, actor_user_id, actor_username, server_id, action, operation_id, "
            "player_id, attempt, detail_json) VALUES "
            "(:created_at, :actor_user_id, :actor_username, :server_id, :action, :operation_id, "
            ":player_id, :attempt, :detail_json)",
            {
                "created_at": created_at,
                "actor_user_id": actor_user_id,
                "actor_username": actor_username,
                "server_id": server_id,
                "action": action,
                "operation_id": operation_id,
                "player_id": player_id,
                "attempt": attempt,
                "detail_json": json_dumps(detail or {}),
            },
        )

    async def _maybe_finish_operation(
        self, db: DatabaseConnection, operation_id: str, now: float
    ) -> Mapping[str, Any] | None:
        """
        Finish the operation within an open transaction when no target is outstanding.

        :return: The operation row when it transitioned to a terminal status, else None.
        """
        cursor = await db.execute(
            f"SELECT COUNT(*) FROM {PC_TABLE_TARGETS} WHERE operation_id = :operation_id "
            f"AND status NOT IN ({_TERMINAL_TARGET_LIST})",
            {"operation_id": operation_id},
        )
        row = await cursor.fetchone()
        if row is not None and int(row[0]) > 0:
            return None
        op_cursor = await db.execute(
            f"SELECT * FROM {PC_TABLE_OPERATIONS} WHERE operation_id = :operation_id",
            {"operation_id": operation_id},
        )
        op_row = await op_cursor.fetchone()
        if op_row is None or op_row["status"] in TERMINAL_OPERATION_STATUSES:
            return None
        counts = await self._counts_in_tx(db, operation_id)
        if op_row["cancel_requested"]:
            new_status = OperationStatus.CANCELLED
        elif counts["succeeded"] == counts["total"]:
            new_status = OperationStatus.SUCCEEDED
        elif counts["succeeded"] > 0:
            new_status = OperationStatus.PARTIAL
        else:
            new_status = OperationStatus.FAILED
        await db.execute(
            f"UPDATE {PC_TABLE_OPERATIONS} SET status = :status, finished_at = :now, "
            "started_at = COALESCE(started_at, :now) WHERE operation_id = :operation_id",
            {"status": new_status.value, "now": now, "operation_id": operation_id},
        )
        await self._insert_event(
            db,
            operation_id=operation_id,
            event_type=BatchEventType.OPERATION_STATUS_CHANGED,
            created_at=now,
            payload={"status": new_status.value},
        )
        await self._insert_event(
            db,
            operation_id=operation_id,
            event_type=BatchEventType.OPERATION_FINISHED,
            created_at=now,
            payload={"status": new_status.value, "counts": dict(counts)},
        )
        await self._insert_audit(
            db,
            created_at=now,
            server_id=op_row["created_server_id"],
            action=AUDIT_OPERATION_FINISHED,
            operation_id=operation_id,
            actor_user_id=op_row["created_by_user_id"],
            actor_username=op_row["created_by_name"],
            detail={"status": new_status.value, "counts": dict(counts)},
        )
        final_cursor = await db.execute(
            f"SELECT * FROM {PC_TABLE_OPERATIONS} WHERE operation_id = :operation_id",
            {"operation_id": operation_id},
        )
        return cast("Mapping[str, Any] | None", await final_cursor.fetchone())

    async def _counts_in_tx(self, db: DatabaseConnection, operation_id: str) -> dict[str, int]:
        cursor = await db.execute(
            f"SELECT status, COUNT(*) FROM {PC_TABLE_TARGETS} "
            "WHERE operation_id = :operation_id GROUP BY status",
            {"operation_id": operation_id},
        )
        counts = {
            "total": 0,
            TargetStatus.PENDING.value: 0,
            TargetStatus.RUNNING.value: 0,
            TargetStatus.RETRY_WAIT.value: 0,
            TargetStatus.SUCCEEDED.value: 0,
            TargetStatus.FAILED.value: 0,
            TargetStatus.CANCELLED.value: 0,
            TargetStatus.UNREACHABLE.value: 0,
            TargetStatus.REPLACED.value: 0,
            TargetStatus.SEALED.value: 0,
        }
        for status, count in await cursor.fetchall():
            counts[str(status)] = int(count)
            counts["total"] += int(count)
        return counts

    async def _open_database(self, db_path: str) -> int:
        if self._database is None:
            database = DatabaseConnection(db_path)
            try:
                await database.setup()
            except BaseException:
                with contextlib.suppress(Exception):
                    await database.close()
                raise
            self._database = database
        try:
            return await create_player_commands_tables(self._database)
        except BaseException:
            await self.close_database()
            raise

    async def _quarantine_database(self, db_path: str) -> None:
        await self.close_database()
        for suffix in ("", "-wal", "-shm"):
            source = f"{db_path}{suffix}"
            if await asyncio.to_thread(os.path.exists, source):
                await asyncio.to_thread(os.replace, source, f"{source}.corrupt")

