"""Client-facing data models for the asynchronous batch player commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mashumaro.mixins.orjson import DataClassORJSONMixin


@dataclass(kw_only=True)
class TargetCounts(DataClassORJSONMixin):
    """Aggregated counts of targets by state within one operation."""

    total: int = 0
    pending: int = 0
    running: int = 0
    retry_wait: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0
    unreachable: int = 0
    replaced: int = 0
    sealed: int = 0

    @property
    def outstanding(self) -> int:
        """Number of targets that have not reached a terminal state."""
        return self.pending + self.running + self.retry_wait


@dataclass(kw_only=True)
class OperationRecord(DataClassORJSONMixin):
    """Public view of a batch operation."""

    operation_id: str
    command: str
    args: dict[str, Any] = field(default_factory=dict)
    status: str
    player_ids: list[str] = field(default_factory=list)
    max_concurrency: int | None = None
    cancel_requested: bool = False
    created_by_user_id: str | None = None
    created_by_name: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    counts: TargetCounts = field(default_factory=TargetCounts)


@dataclass(kw_only=True)
class TargetRecord(DataClassORJSONMixin):
    """Public view of a single target within a batch operation."""

    player_id: str
    status: str
    outcome: str | None = None
    effective_player_id: str | None = None
    provider_instance: str = ""
    provider_domain: str = ""
    attempts: int = 0
    max_attempts: int
    next_run_at: float | None = None
    lease_expires_at: float | None = None
    last_error_code: int | None = None
    last_error_message: str | None = None
    cancel_boundary: bool = False
    created_at: float
    dispatched_at: float | None = None
    finished_at: float | None = None


@dataclass(kw_only=True)
class OperationSnapshot(DataClassORJSONMixin):
    """Full state of one operation: the operation record and all visible targets."""

    operation: OperationRecord
    targets: list[TargetRecord] = field(default_factory=list)
    next_cursor: int = 0


@dataclass(kw_only=True)
class SubmitResult(DataClassORJSONMixin):
    """Result of a submit call; reused=True when an existing operation was returned."""

    snapshot: OperationSnapshot
    reused: bool = False


@dataclass(kw_only=True)
class OperationEvent(DataClassORJSONMixin):
    """One append-only event in a batch operation's event log."""

    cursor: int
    operation_id: str
    event_type: str
    created_at: float
    player_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class EventPage(DataClassORJSONMixin):
    """
    One page of a cursor-based event subscription.

    ``resync`` is set when the requested cursor is older than the retained event
    window (gap) or when the operation no longer exists; the client must rebuild its
    state from the attached snapshot instead of applying the event list.
    """

    events: list[OperationEvent] = field(default_factory=list)
    next_cursor: int
    resync: bool = False
    resync_reason: str | None = None
    snapshot: OperationSnapshot | None = None
    terminal: bool = False
