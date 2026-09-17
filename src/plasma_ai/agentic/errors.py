"""Fail-closed exceptions for Phase 7 trusted boundaries."""

from __future__ import annotations


class TrustedBoundaryError(RuntimeError):
    """Raised when trusted upstream evidence cannot be established."""

    __slots__ = ("boundary", "stage", "cause_type")

    _PERMITTED_BOUNDARIES = frozenset(
        {
            "phase5_monitoring",
            "phase6_optimisation",
        }
    )

    def __init__(
        self,
        *,
        boundary: str,
        stage: str,
        cause_type: str,
    ) -> None:
        """Create a deterministic trusted-boundary failure."""

        if boundary not in self._PERMITTED_BOUNDARIES:
            raise ValueError("Unsupported trusted boundary.")

        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("Trusted-boundary stage must be non-empty.")

        if not isinstance(cause_type, str) or not cause_type.strip():
            raise ValueError("Trusted-boundary cause_type must be non-empty.")

        self.boundary = boundary
        self.stage = stage
        self.cause_type = cause_type

        super().__init__(
            "Trusted boundary failure: "
            f"boundary={boundary}; "
            f"stage={stage}; "
            f"cause_type={cause_type}."
        )
