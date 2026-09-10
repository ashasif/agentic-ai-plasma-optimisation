"""Fault-profile mathematics for Phase-3 synthetic monitoring episodes."""

from __future__ import annotations

import math

from plasma_ai.experiments.monitoring_plan import (
    MonitoringEpisodePlan,
)


def _validate_step_index(
    step_index: int,
    steps_per_episode: int,
) -> None:
    """Validate a zero-based monitoring-step index."""
    if steps_per_episode <= 0:
        raise ValueError(
            "steps_per_episode must be strictly positive."
        )

    if (
        step_index < 0
        or step_index >= steps_per_episode
    ):
        raise ValueError(
            "step_index lies outside the monitoring episode."
        )


def fault_started(
    plan: MonitoringEpisodePlan,
    step_index: int,
    steps_per_episode: int,
) -> bool:
    """Return whether a configured fault has reached its onset step."""
    _validate_step_index(
        step_index,
        steps_per_episode,
    )

    if not plan.fault_present:
        return False

    if plan.fault_onset_step is None:
        raise ValueError(
            "Faulty episodes require fault_onset_step."
        )

    return bool(
        step_index >= plan.fault_onset_step
    )


def fault_progress(
    plan: MonitoringEpisodePlan,
    step_index: int,
    steps_per_episode: int,
) -> float:
    """Return normalized fault development in [0, 1].

    Step faults jump from zero to one at the onset step.

    Drift faults use the frozen cubic smoothstep profile

        g(u) = 3 u^2 - 2 u^3

    from the onset step to the final episode step.

    For a drift, progress is exactly zero at the onset step even
    though ``fault_started`` is already True.
    """
    _validate_step_index(
        step_index,
        steps_per_episode,
    )

    if not plan.fault_present:
        return 0.0

    onset = plan.fault_onset_step

    if onset is None:
        raise ValueError(
            "Faulty episodes require fault_onset_step."
        )

    if (
        onset < 0
        or onset >= steps_per_episode
    ):
        raise ValueError(
            "fault_onset_step lies outside the episode."
        )

    if step_index < onset:
        return 0.0

    if plan.fault_profile == "step":
        return 1.0

    if plan.fault_profile != "drift":
        raise ValueError(
            f"Unknown fault profile: {plan.fault_profile!r}"
        )

    final_step = (
        steps_per_episode
        - 1
    )

    if onset >= final_step:
        raise ValueError(
            "A drift fault must begin before the final episode step."
        )

    u = (
        (step_index - onset)
        / (final_step - onset)
    )

    u = min(
        1.0,
        max(
            0.0,
            float(u),
        ),
    )

    progress = (
        3.0 * u**2
        - 2.0 * u**3
    )

    return float(
        min(
            1.0,
            max(
                0.0,
                progress,
            ),
        )
    )


def signed_fault_fraction(
    plan: MonitoringEpisodePlan,
    step_index: int,
    steps_per_episode: int,
) -> float:
    """Return the signed fractional fault perturbation."""
    if not plan.fault_present:
        _validate_step_index(
            step_index,
            steps_per_episode,
        )
        return 0.0

    magnitude = float(
        plan.fault_magnitude_fraction
    )

    if (
        not math.isfinite(magnitude)
        or magnitude < 0.0
        or magnitude >= 1.0
    ):
        raise ValueError(
            "fault_magnitude_fraction must lie in [0, 1)."
        )

    if plan.fault_direction == "negative":
        sign = -1.0
    elif plan.fault_direction == "positive":
        sign = 1.0
    else:
        raise ValueError(
            f"Unknown fault direction: {plan.fault_direction!r}"
        )

    return float(
        sign
        * magnitude
        * fault_progress(
            plan,
            step_index,
            steps_per_episode,
        )
    )


def multiplicative_fault_factor(
    plan: MonitoringEpisodePlan,
    step_index: int,
    steps_per_episode: int,
) -> float:
    """Return 1 + the signed fractional fault perturbation."""
    factor = (
        1.0
        + signed_fault_fraction(
            plan,
            step_index,
            steps_per_episode,
        )
    )

    if (
        not math.isfinite(factor)
        or factor <= 0.0
    ):
        raise ValueError(
            "Fault factor must remain finite and strictly positive."
        )

    return float(
        factor
    )
