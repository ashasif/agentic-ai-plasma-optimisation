"""Tests for the Phase 7 trusted-boundary error contract."""

from __future__ import annotations

import pytest

from plasma_ai.agentic.errors import TrustedBoundaryError


def _wrap_ordinary_exception(callback) -> None:
    try:
        callback()
    except Exception as exc:
        raise TrustedBoundaryError(
            boundary="phase5_monitoring",
            stage="prediction",
            cause_type=type(exc).__name__,
        ) from exc


def test_trusted_boundary_error_is_runtime_error() -> None:
    error = TrustedBoundaryError(
        boundary="phase5_monitoring",
        stage="load",
        cause_type="ValueError",
    )

    assert isinstance(error, RuntimeError)


def test_boundary_stage_and_cause_type_are_preserved() -> None:
    error = TrustedBoundaryError(
        boundary="phase6_optimisation",
        stage="execution",
        cause_type="ProductionRuntimeError",
    )

    assert error.boundary == "phase6_optimisation"
    assert error.stage == "execution"
    assert error.cause_type == "ProductionRuntimeError"


def test_ordinary_exception_is_wrapped_and_chained() -> None:
    def fail() -> None:
        raise ValueError("synthetic test failure")

    with pytest.raises(TrustedBoundaryError) as captured:
        _wrap_ordinary_exception(fail)

    assert captured.value.boundary == "phase5_monitoring"
    assert captured.value.stage == "prediction"
    assert captured.value.cause_type == "ValueError"
    assert isinstance(captured.value.__cause__, ValueError)


def test_keyboard_interrupt_is_not_caught() -> None:
    def interrupt() -> None:
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        _wrap_ordinary_exception(interrupt)


def test_system_exit_is_not_caught() -> None:
    def exit_now() -> None:
        raise SystemExit(7)

    with pytest.raises(SystemExit) as captured:
        _wrap_ordinary_exception(exit_now)

    assert captured.value.code == 7


@pytest.mark.parametrize(
    "boundary",
    [
        "phase5_monitoring",
        "phase6_optimisation",
    ],
)
def test_permitted_boundary_vocabulary(boundary: str) -> None:
    error = TrustedBoundaryError(
        boundary=boundary,
        stage="load",
        cause_type="RuntimeError",
    )

    assert error.boundary == boundary


def test_unknown_boundary_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported trusted boundary"):
        TrustedBoundaryError(
            boundary="unknown",
            stage="load",
            cause_type="RuntimeError",
        )


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_stage_is_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        TrustedBoundaryError(
            boundary="phase5_monitoring",
            stage=value,
            cause_type="RuntimeError",
        )


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_cause_type_is_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        TrustedBoundaryError(
            boundary="phase5_monitoring",
            stage="load",
            cause_type=value,
        )
