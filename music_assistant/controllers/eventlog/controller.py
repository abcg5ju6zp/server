"""Persistent event log backing resumable WebSocket subscriptions."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import hashlib
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import EventType
from music_assistant_models.event import MassEvent

from music_assistant.helpers.database import DatabaseConnection
from music_assistant.helpers.json import json_dumps, json_loads
from music_assistant.models.core_controller import CoreController

from .constants import (
    DB_SCHEMA_VERSION,
    DB_TABLE_EVENTS,
    DB_TABLE_SETTINGS,
    DB_TABLE_SUBSCRIPTIONS,
    FLUSH_BATCH_SIZE,
    FLUSH_INTERVAL,
    LOGGER,
    MAX_WINDOW_AGE_SEC,
    MAX_WINDOW_EVENTS,
    PRUNE_EVERY_N_INSERTS,
    SUBSCRIPTION_TTL_SEC,
)
from .models import ResumeMode, SnapshotReason, freeze_payload, rehydrate_payload

if TYPE_CHECKING:
    from music_assistant_models.auth import User
    from music_assistant_models.config_entries import CoreConfig

    from music_assistant import MusicAssistant


class EventWindowTrimmedError(Exception):
    """Raised when a cursor points before the oldest event still held in the window."""


@dataclass(slots=True)
class EventRecord:
    """One admitted event with its global sequence and frozen payload."""

    seq: int
    event: str
    object_id: str | None
    ts: float
    data_type: str
    data: Any
    cached_event: MassEvent | None = field(default=None, repr=False)


@dataclass(slots=True, eq=False)
class ResumableSubscription:
    """
    Per-connection state of a resumable event stream.

    ``buffering`` is on between listener registration and the end of the subscribe
    request: live events that arrive while the request awaits land in ``buffer`` so
    they can be delivered after the response, in their original sequence.

    :param client: The websocket client handler owning this subscription.
    """

    client: Any
    buffering: bool = True
    buffer: deque[EventRecord] = field(default_factory=deque)
    # highest sequence handed to (or filtered out for) this connection, contiguously
    sent_watermark: int = 0
    # highest contiguous sequence this connection has acknowledged
    acked_watermark: int = 0
    token: str | None = None
    user_id: str | None = None
    scope_sig: str | None = None

    def on_record(self, record: EventRecord) -> None:
        """Receive an admitted record, buffering it until the subscription is armed."""
        if self.buffering:
            self.buffer.append(record)
        else:
            self.client._deliver_event_record(self, record)


def compute_scope_sig(user: User) -> str:
    """
    Hash the permission boundary of a user (role and access filters).

    A reconnect presenting a cursor must bind to the same signature; otherwise the
    replay would bridge states assembled under different permissions.

    :param user: The authenticated user to fingerprint.
    """
    payload = json_dumps(
        [
            user.role,
            sorted(user.provider_filter or []),
            sorted(user.player_filter or []),
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class EventLogController(CoreController):
    """
    Assign a monotonic cursor to every event and keep a bounded persistent window.

    Every signalled event is frozen (type, object id and payload) and sequenced on the
    event loop thread before anything is broadcast, so live fan-out and replay share
    one immutable record. Rows survive a restart in ``eventlog.db``; this is purely
    operational replay data (like ``cache.db``), never user data, so an unreadable or
    unknown-schema database is reset instead of migrated.
    """

    domain: str = "event_log"

    def __init__(self, mass: MusicAssistant) -> None:
        """Initialize core controller."""
        super().__init__(mass)
        self.database: DatabaseConnection | None = None
        self._records: deque[EventRecord] = deque()
        self._listeners: set[ResumableSubscription] = set()
        self._write_q: asyncio.Queue[tuple[Any, ...] | None] = asyncio.Queue()
        self._writer_task: asyncio.Task[None] | None = None
        self._head_seq = 0
        self._inserts_since_prune = 0
        self._ready = False
        self.manifest.name = "Event log controller"
        self.manifest.description = (
            "Music Assistant's core controller for the resumable event stream."
        )
        self.manifest.icon = "pulse"

    @property
    def ready(self) -> bool:
        """Return whether the event log finished setting up and accepts events."""
        return self._ready

    @property
    def head_seq(self) -> int:
        """Return the highest sequence assigned so far."""
        return self._head_seq

    @property
    def oldest_seq(self) -> int:
        """Return the oldest sequence still available for replay."""
        return self._records[0].seq if self._records else self._head_seq + 1

    async def setup(self, config: CoreConfig) -> None:
        """Async initialize of module."""
        db_path = os.path.join(self.mass.storage_path, "eventlog.db")
        self.database = DatabaseConnection(db_path)
        await self.database.setup()
        try:
            await self._create_tables()
            version = await self._get_schema_version()
            if version not in (0, DB_SCHEMA_VERSION):
                # version 0 is a fresh database (no row yet); anything else but the
                # current version is unsupported operational data and gets reset
                LOGGER.warning(
                    "Event log database schema version %s is unsupported (expected %s), "
                    "resetting event log",
                    version,
                    DB_SCHEMA_VERSION,
                )
                await self._reset_database()
            await self._load_window()
        except Exception as err:
            # never block server startup for a replay cache: start with an empty window
            LOGGER.warning("Event log database unusable, resetting: %s", str(err))
            await self._reset_database()
        # intentionally NOT tracked via mass.create_task: stop() cancels every tracked
        # task before closing controllers, but this writer must still be able to flush
        # when close() runs
        self._writer_task = asyncio.Task(
            self._writer(), loop=self.mass.loop, eager_start=False, name="eventlog-writer"
        )
        self._ready = True
        self.logger.info("Event log initialized (head seq: %s)", self._head_seq)

    async def close(self) -> None:
        """Cleanup on exit."""
        self._ready = False
        # tell the writer to flush everything pending and exit, then close the db
        await self._write_q.put(None)
        if self._writer_task is not None:
            try:
                await asyncio.wait_for(self._writer_task, timeout=10)
            except TimeoutError, asyncio.CancelledError:
                self._writer_task.cancel()
        if self.database is not None:
            # the connection may be half-initialized after a failed setup
            with contextlib.suppress(Exception):
                await self.database.close()
            self.database = None

    def admit(self, event: MassEvent) -> EventRecord | None:
        """
        Freeze and sequence an event, persist it and fan it out to live listeners.

        Must run on the event loop thread, before the regular ``signal_event`` fan-out:
        the assigned order therefore equals the broadcast order. Any failure here is
        logged and swallowed so event dispatch itself never breaks.

        :param event: The freshly signalled event.
        """
        if not self._ready:
            return None
        self.mass.verify_event_loop_thread("eventlog.admit")
        seq = self._head_seq + 1
        try:
            data_type, frozen_data = contextvars.Context().run(freeze_payload, event.data)
        except Exception as err:
            LOGGER.warning("Failed to freeze event %s (seq %s): %s", event.event, seq, str(err))
            return None
        record = EventRecord(
            seq=seq,
            event=event.event.value,
            object_id=event.object_id,
            ts=time.time(),
            data_type=data_type,
            data=frozen_data,
        )
        self._head_seq = seq
        self._records.append(record)
        self._write_q.put_nowait(("event", record))
        self._inserts_since_prune += 1
        if self._inserts_since_prune >= PRUNE_EVERY_N_INSERTS:
            self._inserts_since_prune = 0
            self._apply_retention()
        for listener in list(self._listeners):
            self.mass.loop.call_soon(listener.on_record, record)
        return record

    def register_listener(self, subscription: ResumableSubscription) -> int:
        """
        Start buffering live events for a connection and return the current head.

        Registration and head capture are one synchronous step on the loop thread, so
        every event admitted afterwards reaches the buffer and every earlier event is
        already readable from the window.

        :param subscription: The connection subscription to register.
        """
        self.mass.verify_event_loop_thread("eventlog.register_listener")
        self._listeners.add(subscription)
        return self._head_seq

    def unregister_listener(self, subscription: ResumableSubscription) -> None:
        """
        Drop a disconnected subscription and release its buffer.

        :param subscription: The connection subscription to remove.
        """
        self._listeners.discard(subscription)
        subscription.buffer.clear()

    def iter_window(self, after_seq: int) -> list[EventRecord]:
        """
        Return stored records with a sequence greater than ``after_seq``.

        :param after_seq: The client's last cursor.
        :raises EventWindowTrimmedError: If the window no longer starts at/before it.
        """
        if after_seq > self._head_seq:
            msg = f"Cursor {after_seq} is ahead of head {self._head_seq}"
            raise EventWindowTrimmedError(msg)
        if self._records and after_seq < self._records[0].seq - 1:
            msg = f"Cursor {after_seq} predates window start {self._records[0].seq}"
            raise EventWindowTrimmedError(msg)
        return [record for record in self._records if record.seq > after_seq]

    def rehydrate_record(self, record: EventRecord) -> MassEvent:
        """
        Rebuild the typed MassEvent behind a record (cached after the first rebuild).

        Both the live and the replay path use this, so they run the same payload
        through the same per-connection filter.

        :param record: The stored event record.
        """
        if record.cached_event is None:
            data = rehydrate_payload(record.data_type, record.data)
            record.cached_event = MassEvent(
                event=EventType(record.event),
                object_id=record.object_id,
                data=data,
            )
        return record.cached_event

    async def validate_resume(
        self,
        token: str | None,
        seq: Any,
        server_id: str | None,
        user: User,
        scope_sig: str,
        head: int,
    ) -> tuple[ResumeMode, SnapshotReason]:
        """
        Decide whether a resume request can be served from the window.

        Any presented token is consumed (tokens are single use and rotated on every
        subscription). A foreign server instance, an unknown token or an out of range
        cursor yields ``INVALID_CURSOR``; a principal/scope mismatch yields
        ``SCOPE_CHANGED``; a trimmed cursor yields ``TRIMMED``.

        :param token: The resume token returned by the previous subscription.
        :param seq: The client's last cursor.
        :param server_id: The server id the client got the cursor from.
        :param user: The authenticated user of the new connection.
        :param scope_sig: The current permission fingerprint of that user.
        :param head: The head captured together with listener registration.
        """
        if not token or not isinstance(seq, int) or isinstance(seq, bool) or seq <= 0:
            return ResumeMode.SNAPSHOT, SnapshotReason.FRESH
        if not isinstance(server_id, str) or server_id != self.mass.server_id:
            return ResumeMode.SNAPSHOT, SnapshotReason.INVALID_CURSOR
        assert self.database is not None
        row = await self.database.get_row(DB_TABLE_SUBSCRIPTIONS, {"token": token})
        if row is not None:
            # rotate: a token is valid for a single resume, whichever decision follows
            await self.database.delete(DB_TABLE_SUBSCRIPTIONS, {"token": token})
        if row is None:
            return ResumeMode.SNAPSHOT, SnapshotReason.INVALID_CURSOR
        if row["user_id"] != user.user_id or row["scope_sig"] != scope_sig:
            return ResumeMode.SNAPSHOT, SnapshotReason.SCOPE_CHANGED
        if seq > head:
            return ResumeMode.SNAPSHOT, SnapshotReason.INVALID_CURSOR
        if self._records and seq < self._records[0].seq - 1:
            return ResumeMode.SNAPSHOT, SnapshotReason.TRIMMED
        return ResumeMode.RESUMED, SnapshotReason.FRESH

    async def create_subscription(self, user: User, scope_sig: str) -> str:
        """
        Persist and return a fresh single-use resume token for a user/scope pair.

        :param user: The authenticated user the token binds to.
        :param scope_sig: The permission fingerprint the token binds to.
        """
        assert self.database is not None
        token = uuid.uuid4().hex
        await self.database.insert(
            DB_TABLE_SUBSCRIPTIONS,
            {
                "token": token,
                "user_id": user.user_id,
                "scope_sig": scope_sig,
                "created_ts": time.time(),
            },
        )
        return token

    async def update_ack(self, token: str, seq: int) -> None:
        """
        Persist the last acknowledged cursor of a subscription.

        :param token: The token of the connection making the acknowledgement.
        :param seq: The (already clamped) acknowledged watermark.
        """
        assert self.database is not None
        await self.database.update(
            DB_TABLE_SUBSCRIPTIONS,
            {"token": token},
            {"last_ack": seq},
        )

    def _apply_retention(self) -> None:
        """Evict expired/over-capacity/fully-acked records and schedule a db prune."""
        now = time.time()
        ts_cutoff = now - MAX_WINDOW_AGE_SEC
        capacity_floor = self._head_seq - MAX_WINDOW_EVENTS
        # only watermarks of fully armed subscriptions count: a subscription still in
        # its subscribe request has confirmed nothing
        ack_floor = min(
            (
                listener.acked_watermark
                for listener in self._listeners
                if listener.token is not None
            ),
            default=None,
        )
        while self._records:
            record = self._records[0]
            if record.seq > capacity_floor and record.ts >= ts_cutoff:
                if ack_floor is None or record.seq > ack_floor:
                    break
            self._records.popleft()
        oldest_kept = self._records[0].seq if self._records else self._head_seq + 1
        self._write_q.put_nowait(("prune", oldest_kept, ts_cutoff))

    async def _writer(self) -> None:
        """Persist queued records in batches, flushing at least once per interval."""
        pending: list[tuple[Any, ...]] = []
        while True:
            try:
                item = await asyncio.wait_for(self._write_q.get(), timeout=FLUSH_INTERVAL)
            except TimeoutError:
                await self._flush(pending)
                pending = []
                continue
            if item is None:
                await self._flush(pending)
                return
            pending.append(item)
            while len(pending) < FLUSH_BATCH_SIZE:
                try:
                    extra = self._write_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if extra is None:
                    await self._flush(pending)
                    return
                pending.append(extra)
            await self._flush(pending)
            pending = []

    async def _flush(self, ops: list[tuple[Any, ...]]) -> None:
        """Write one batch of event inserts and the latest pruning op to the database."""
        if not ops or self.database is None:
            return
        events = [op[1] for op in ops if op[0] == "event"]
        try:
            if events:
                await self.database.upsert_many(
                    DB_TABLE_EVENTS,
                    [
                        {
                            "seq": record.seq,
                            "event": record.event,
                            "object_id": record.object_id,
                            "ts": record.ts,
                            "data_type": record.data_type,
                            "data": json_dumps(record.data),
                        }
                        for record in events
                    ],
                )
            prunes = [op for op in ops if op[0] == "prune"]
            if prunes:
                _, oldest_kept, ts_cutoff = prunes[-1]
                await self.database.execute_write(
                    f"DELETE FROM {DB_TABLE_EVENTS} WHERE seq < :oldest_kept OR ts < :ts_cutoff",
                    {"oldest_kept": oldest_kept, "ts_cutoff": ts_cutoff},
                )
                await self.database.execute_write(
                    f"DELETE FROM {DB_TABLE_SUBSCRIPTIONS} WHERE created_ts < :cutoff",
                    {"cutoff": time.time() - SUBSCRIPTION_TTL_SEC},
                )
        except Exception as err:
            LOGGER.warning("Failed to flush event log batch: %s", str(err))

    async def _get_schema_version(self) -> int:
        """Return the schema version stored in the settings table."""
        assert self.database is not None
        if db_row := await self.database.get_row(DB_TABLE_SETTINGS, {"key": "version"}):
            return int(db_row["value"])
        return 0

    async def _create_tables(self) -> None:
        """Create the event log tables if they do not exist."""
        assert self.database is not None
        await self.database.execute(
            f"""CREATE TABLE IF NOT EXISTS {DB_TABLE_SETTINGS}(
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    type TEXT
                );"""
        )
        await self.database.execute(
            f"""CREATE TABLE IF NOT EXISTS {DB_TABLE_EVENTS}(
                    [seq] INTEGER PRIMARY KEY,
                    [event] TEXT NOT NULL,
                    [object_id] TEXT,
                    [ts] REAL NOT NULL,
                    [data_type] TEXT NOT NULL,
                    [data] TEXT NOT NULL
                );"""
        )
        await self.database.execute(
            f"""CREATE TABLE IF NOT EXISTS {DB_TABLE_SUBSCRIPTIONS}(
                    [token] TEXT PRIMARY KEY,
                    [user_id] TEXT NOT NULL,
                    [scope_sig] TEXT NOT NULL,
                    [created_ts] REAL NOT NULL,
                    [last_ack] INTEGER NOT NULL DEFAULT 0
                );"""
        )
        await self.database.commit()

    async def _reset_database(self) -> None:
        """Drop and recreate all tables, discarding the operational replay data."""
        assert self.database is not None
        await self.database.execute(f"DROP TABLE IF EXISTS {DB_TABLE_EVENTS}")
        await self.database.execute(f"DROP TABLE IF EXISTS {DB_TABLE_SUBSCRIPTIONS}")
        await self.database.execute(f"DROP TABLE IF EXISTS {DB_TABLE_SETTINGS}")
        await self.database.commit()
        await self._create_tables()
        self._records.clear()
        self._head_seq = 0
        await self.database.insert_or_replace(
            DB_TABLE_SETTINGS,
            {"key": "version", "value": str(DB_SCHEMA_VERSION), "type": "str"},
        )

    async def _load_window(self) -> None:
        """Load the persistent window back into memory, continuing the sequence."""
        assert self.database is not None
        await self.database.insert_or_replace(
            DB_TABLE_SETTINGS,
            {"key": "version", "value": str(DB_SCHEMA_VERSION), "type": "str"},
        )
        rows = await self.database.get_rows_from_query(
            f"SELECT seq, event, object_id, ts, data_type, data FROM {DB_TABLE_EVENTS} "
            "ORDER BY seq DESC LIMIT :limit",
            {"limit": MAX_WINDOW_EVENTS},
            limit=0,
        )
        age_cutoff = time.time() - MAX_WINDOW_AGE_SEC
        records: deque[EventRecord] = deque()
        expected_seq: int | None = None
        for row in rows:  # rows come back newest first
            seq = int(row["seq"])
            if expected_seq is None:
                expected_seq = seq
                self._head_seq = seq
            # keep only the contiguous suffix ending at the head: a missing seq (e.g.
            # a crash before the writer flushed) invalidates everything older
            if seq != expected_seq or float(row["ts"]) < age_cutoff:
                break
            records.appendleft(
                EventRecord(
                    seq=seq,
                    event=str(row["event"]),
                    object_id=row["object_id"],
                    ts=float(row["ts"]),
                    data_type=str(row["data_type"]),
                    data=json_loads(row["data"]),
                )
            )
            expected_seq -= 1
        self._records = records
