"""Deterministic random utilities for Phase-3 monitoring data."""

from __future__ import annotations

import hashlib
import math

import numpy as np


_RANDOM_NAMESPACE = "phase3_monitoring_v1"


def deterministic_rng(
    base_seed: int,
    episode_id: str,
    step_index: int,
    channel: str,
) -> np.random.Generator:
    """Return an independently derived deterministic RNG stream.

    The stream depends only on the supplied identifiers, so results do
    not depend on generation order or on other monitoring channels.
    """
    if base_seed < 0:
        raise ValueError(
            "base_seed must be non-negative."
        )

    if not episode_id:
        raise ValueError(
            "episode_id must be non-empty."
        )

    if step_index < 0:
        raise ValueError(
            "step_index must be non-negative."
        )

    if not channel:
        raise ValueError(
            "channel must be non-empty."
        )

    payload = (
        f"{_RANDOM_NAMESPACE}|"
        f"{base_seed}|"
        f"{episode_id}|"
        f"{step_index}|"
        f"{channel}"
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        payload
    ).digest()

    derived_seed = int.from_bytes(
        digest[:16],
        byteorder="big",
        signed=False,
    )

    return np.random.default_rng(
        derived_seed
    )


def truncated_standard_normal(
    rng: np.random.Generator,
    truncation_sigma: float,
) -> float:
    """Draw from N(0,1) conditioned to lie within ?truncation_sigma."""
    if (
        not math.isfinite(truncation_sigma)
        or truncation_sigma <= 0.0
    ):
        raise ValueError(
            "truncation_sigma must be finite and strictly positive."
        )

    while True:
        value = float(
            rng.normal()
        )

        if abs(value) <= truncation_sigma:
            return value


def deterministic_relative_factor(
    *,
    base_seed: int,
    episode_id: str,
    step_index: int,
    channel: str,
    relative_sigma: float,
    truncation_sigma: float,
) -> float:
    """Return a deterministic multiplicative factor 1 + epsilon.

    epsilon is sampled from a zero-mean Gaussian with the configured
    relative standard deviation and truncated at ?truncation_sigma.
    """
    if (
        not math.isfinite(relative_sigma)
        or relative_sigma < 0.0
        or relative_sigma >= 1.0
    ):
        raise ValueError(
            "relative_sigma must be finite and lie in [0, 1)."
        )

    if (
        not math.isfinite(truncation_sigma)
        or truncation_sigma <= 0.0
    ):
        raise ValueError(
            "truncation_sigma must be finite and strictly positive."
        )

    if (
        relative_sigma
        * truncation_sigma
        >= 1.0
    ):
        raise ValueError(
            "Configured truncation could generate a non-positive factor."
        )

    if relative_sigma == 0.0:
        return 1.0

    rng = deterministic_rng(
        base_seed=base_seed,
        episode_id=episode_id,
        step_index=step_index,
        channel=channel,
    )

    z_value = truncated_standard_normal(
        rng,
        truncation_sigma,
    )

    factor = (
        1.0
        + relative_sigma
        * z_value
    )

    if (
        not math.isfinite(factor)
        or factor <= 0.0
    ):
        raise RuntimeError(
            "Generated relative factor is not physically positive."
        )

    return float(
        factor
    )
