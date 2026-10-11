"""Errors for the asynchronous batch player commands controller."""

from __future__ import annotations

from music_assistant_models.errors import InvalidDataError


class BatchOperationConflictError(InvalidDataError):
    """
    Raised when a submit request reuses an operation_id with a different payload.

    The client chose a stable operation_id that already identifies an operation with a
    different command, argument set, target set or concurrency limit. The existing
    operation is left untouched; the caller must pick a new id or resubmit the
    identical payload to idempotently retrieve the original operation.
    """

    def __init__(self, operation_id: str, detail: str) -> None:
        """Raise the conflict for operation_id describing how the payloads differ."""
        super().__init__(
            f"Operation {operation_id} already exists with a different payload: {detail}",
            translation_key="player_commands/operation_conflict",
            translation_args=[operation_id],
        )
        self.operation_id = operation_id
