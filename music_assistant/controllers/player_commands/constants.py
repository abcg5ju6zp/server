"""Constants and enums for the asynchronous batch player commands controller."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

# database
PC_DB_FILENAME: Final[str] = "player_commands.db"
PC_DB_SCHEMA_VERSION: Final[int] = 1
PC_TABLE_SETTINGS: Final[str] = "settings"
PC_TABLE_OPERATIONS: Final[str] = "batch_operations"
PC_TABLE_TARGETS: Final[str] = "batch_targets"
PC_TABLE_EVENTS: Final[str] = "batch_events"
PC_TABLE_AUDIT: Final[str] = "batch_audit"

# IMPORTANT: player_commands.db holds operational data only and starts at schema v1.
# It is never mixed with library.db, so a failed migration here can never reset the
# music library. Any future schema change must bump PC_DB_SCHEMA_VERSION, add an
# idempotent step to migrations.py (survive missing/half-written rows, never raise),
# and be agreed with the developer in the session before the work is considered done.

# request limits
MAX_TARGETS_PER_OPERATION: Final[int] = 200
MAX_OPERATION_ID_LENGTH: Final[int] = 64
MAX_EVENT_PAGE_LIMIT: Final[int] = 1000
DEFAULT_EVENT_PAGE_LIMIT: Final[int] = 200
DEFAULT_EVENT_WAIT_TIMEOUT: Final[float] = 25.0


class OperationStatus(StrEnum):
    """Lifecycle status of a batch player command operation."""

    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    # terminal states
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_OPERATION_STATUSES: Final[frozenset[OperationStatus]] = frozenset(
    {
        OperationStatus.SUCCEEDED,
        OperationStatus.PARTIAL,
        OperationStatus.FAILED,
        OperationStatus.CANCELLED,
    }
)


class TargetStatus(StrEnum):
    """Per-target state within a batch player command operation."""

    PENDING = "pending"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    # terminal states
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    # the player was permanently removed or never came back before the grace window
    UNREACHABLE = "unreachable"
    # the same player id is now backed by a different player/provider identity
    REPLACED = "replaced"
    # startup reconciliation could never resolve the target after providers loaded
    SEALED = "sealed"


TERMINAL_TARGET_STATUSES: Final[frozenset[TargetStatus]] = frozenset(
    {
        TargetStatus.SUCCEEDED,
        TargetStatus.FAILED,
        TargetStatus.CANCELLED,
        TargetStatus.UNREACHABLE,
        TargetStatus.REPLACED,
        TargetStatus.SEALED,
    }
)


class TargetOutcome(StrEnum):
    """How a successful target was actually executed."""

    DIRECT = "direct"
    REDIRECTED = "redirected"


class BatchEventType(StrEnum):
    """Types of events persisted on the append-only batch event log."""

    OPERATION_CREATED = "operation_created"
    OPERATION_STATUS_CHANGED = "operation_status_changed"
    TARGET_DISPATCHED = "target_dispatched"
    TARGET_RETRYING = "target_retrying"
    TARGET_SUCCEEDED = "target_succeeded"
    TARGET_FAILED = "target_failed"
    TARGET_CANCELLED = "target_cancelled"
    TARGET_UNREACHABLE = "target_unreachable"
    TARGET_REPLACED = "target_replaced"
    TARGET_SEALED = "target_sealed"
    CANCEL_BOUNDARY = "cancel_boundary"
    OPERATION_FINISHED = "operation_finished"


# maps a terminal target status to the event type that announces it
TERMINAL_TARGET_EVENT: Final[dict[TargetStatus, BatchEventType]] = {
    TargetStatus.SUCCEEDED: BatchEventType.TARGET_SUCCEEDED,
    TargetStatus.FAILED: BatchEventType.TARGET_FAILED,
    TargetStatus.CANCELLED: BatchEventType.TARGET_CANCELLED,
    TargetStatus.UNREACHABLE: BatchEventType.TARGET_UNREACHABLE,
    TargetStatus.REPLACED: BatchEventType.TARGET_REPLACED,
    TargetStatus.SEALED: BatchEventType.TARGET_SEALED,
}


# audit actions
AUDIT_OPERATION_CREATED: Final[str] = "operation_created"
AUDIT_OPERATION_CANCELLED: Final[str] = "operation_cancelled"
AUDIT_TARGET_DISPATCHED: Final[str] = "target_dispatched"
AUDIT_OPERATION_FINISHED: Final[str] = "operation_finished"


# normalized error codes (mirror music_assistant_models.errors error_code values,
# plus local codes for the conditions that do not map to a provider error)
ERROR_PLAYER_UNAVAILABLE: Final[int] = 10
ERROR_INSUFFICIENT_PERMISSIONS: Final[int] = 22
ERROR_PLAYER_NOT_FOUND: Final[int] = 1001
ERROR_PROVIDER_INSTANCE_CHANGED: Final[int] = 1002
ERROR_TARGET_SEALED: Final[int] = 1003
ERROR_CANCELLED: Final[int] = 1004


# config keys
CONF_PC_MAX_GLOBAL_CONCURRENCY: Final[str] = "max_global_concurrency"
CONF_PC_MAX_PROVIDER_CONCURRENCY: Final[str] = "max_provider_concurrency"
CONF_PC_DEFAULT_OPERATION_CONCURRENCY: Final[str] = "default_operation_concurrency"
CONF_PC_COMMAND_TIMEOUT: Final[str] = "command_timeout_seconds"
CONF_PC_LEASE_GRACE: Final[str] = "lease_grace_seconds"
CONF_PC_MAX_ATTEMPTS: Final[str] = "max_attempts"
CONF_PC_RETRY_BASE: Final[str] = "retry_base_seconds"
CONF_PC_RETRY_CAP: Final[str] = "retry_cap_seconds"
CONF_PC_PROVIDER_GRACE: Final[str] = "provider_grace_seconds"
CONF_PC_RETENTION_DAYS: Final[str] = "retention_days"
CONF_PC_EVENT_GRACE_HOURS: Final[str] = "event_grace_hours"

DEFAULT_CONFIG: Final[dict[str, int | float]] = {
    CONF_PC_MAX_GLOBAL_CONCURRENCY: 16,
    CONF_PC_MAX_PROVIDER_CONCURRENCY: 4,
    CONF_PC_DEFAULT_OPERATION_CONCURRENCY: 8,
    CONF_PC_COMMAND_TIMEOUT: 30,
    CONF_PC_LEASE_GRACE: 15,
    CONF_PC_MAX_ATTEMPTS: 3,
    CONF_PC_RETRY_BASE: 1.0,
    CONF_PC_RETRY_CAP: 60,
    CONF_PC_PROVIDER_GRACE: 300,
    CONF_PC_RETENTION_DAYS: 7,
    CONF_PC_EVENT_GRACE_HOURS: 24,
}
