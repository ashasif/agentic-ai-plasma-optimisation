"""Deterministic Phase 4E physics-aware acceptance primitives.

These utilities implement the approved pre-test acceptance amendment.

All trend and local-oscillation checks are source-referenced. The functions
operate only on already-evaluated source and surrogate arrays; they do not
execute the plasma simulator, fit models, load TEST targets, or tune criteria.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class PositivityCheckResult:
    """Finite/strict-positivity result for one surrogate target."""

    total_points: int
    nonfinite_count: int
    nonpositive_count: int

    @property
    def passed(self) -> bool:
        """Return whether every prediction is finite and strictly positive."""

        return (
            self.nonfinite_count == 0
            and self.nonpositive_count == 0
        )


@dataclass(frozen=True)
class TrendCheckResult:
    """Source-referenced adjacent-increment trend comparison."""

    axis: int
    total_intervals: int
    unsupported_reversal_count: int
    unsupported_reversal_locations: tuple[
        tuple[int, int],
        ...,
    ]

    @property
    def passed(self) -> bool:
        """Return whether no unsupported trend reversal was found."""

        return self.unsupported_reversal_count == 0


@dataclass(frozen=True)
class OscillationCheckResult:
    """Source-referenced strict local-turning-point comparison."""

    axis: int
    source_turning_point_count: int
    surrogate_turning_point_count: int
    spurious_turning_point_count: int
    spurious_turning_point_locations: tuple[
        tuple[int, int],
        ...,
    ]

    @property
    def passed(self) -> bool:
        """Return whether no source-unsupported turning point was found."""

        return self.spurious_turning_point_count == 0


def _as_finite_shape_matched_arrays(
    source: ArrayLike,
    surrogate: ArrayLike,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return shape-matched float64 source and surrogate arrays."""

    source_array = np.asarray(
        source,
        dtype=np.float64,
    )
    surrogate_array = np.asarray(
        surrogate,
        dtype=np.float64,
    )

    if source_array.ndim != 2:
        raise ValueError(
            "Physics trend diagnostics require 2-D grid arrays."
        )

    if surrogate_array.shape != source_array.shape:
        raise ValueError(
            "Source and surrogate diagnostic arrays must have "
            "identical shapes."
        )

    if not np.all(
        np.isfinite(source_array)
    ):
        raise ValueError(
            "Source diagnostic array contains non-finite values."
        )

    if not np.all(
        np.isfinite(surrogate_array)
    ):
        raise ValueError(
            "Surrogate diagnostic array contains non-finite values."
        )

    return (
        source_array,
        surrogate_array,
    )


def check_strict_positivity(
    values: ArrayLike,
) -> PositivityCheckResult:
    """Apply the approved finite and strict-positivity requirement."""

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    finite = np.isfinite(
        array
    )

    nonfinite_count = int(
        np.count_nonzero(
            ~finite
        )
    )

    nonpositive_count = int(
        np.count_nonzero(
            finite
            & (array <= 0.0)
        )
    )

    return PositivityCheckResult(
        total_points=int(
            array.size
        ),
        nonfinite_count=nonfinite_count,
        nonpositive_count=nonpositive_count,
    )


def compare_source_referenced_trend(
    source: ArrayLike,
    surrogate: ArrayLike,
    *,
    axis: int,
) -> TrendCheckResult:
    """Count surrogate increments strictly opposite to source increments.

    A source zero increment imposes no directional requirement.
    A surrogate zero increment is not an unsupported reversal.
    """

    if axis not in {
        0,
        1,
    }:
        raise ValueError(
            "Phase 4E trend axis must be 0 or 1."
        )

    (
        source_array,
        surrogate_array,
    ) = _as_finite_shape_matched_arrays(
        source,
        surrogate,
    )

    source_delta = np.diff(
        source_array,
        axis=axis,
    )

    surrogate_delta = np.diff(
        surrogate_array,
        axis=axis,
    )

    unsupported = (
        (
            (source_delta > 0.0)
            & (surrogate_delta < 0.0)
        )
        |
        (
            (source_delta < 0.0)
            & (surrogate_delta > 0.0)
        )
    )

    locations = tuple(
        (
            int(location[0]),
            int(location[1]),
        )
        for location in np.argwhere(
            unsupported
        )
    )

    return TrendCheckResult(
        axis=axis,
        total_intervals=int(
            source_delta.size
        ),
        unsupported_reversal_count=int(
            np.count_nonzero(
                unsupported
            )
        ),
        unsupported_reversal_locations=locations,
    )


def _strict_turning_points(
    values: NDArray[np.float64],
    *,
    axis: int,
) -> NDArray[np.bool_]:
    """Return strict local-turning-point mask for a 2-D grid."""

    delta = np.diff(
        values,
        axis=axis,
    )

    if axis == 0:
        return (
            delta[:-1, :]
            * delta[1:, :]
            < 0.0
        )

    return (
        delta[:, :-1]
        * delta[:, 1:]
        < 0.0
    )


def _interior_grid_locations(
    mask: NDArray[np.bool_],
    *,
    axis: int,
) -> tuple[
    tuple[int, int],
    ...,
]:
    """Map turning-point mask positions to original grid coordinates."""

    locations = []

    for location in np.argwhere(
        mask
    ):
        i = int(
            location[0]
        )
        j = int(
            location[1]
        )

        if axis == 0:
            i += 1
        else:
            j += 1

        locations.append(
            (
                i,
                j,
            )
        )

    return tuple(
        locations
    )


def compare_source_referenced_oscillation(
    source: ArrayLike,
    surrogate: ArrayLike,
    *,
    axis: int,
) -> OscillationCheckResult:
    """Detect surrogate strict turning points unsupported by the source."""

    if axis not in {
        0,
        1,
    }:
        raise ValueError(
            "Phase 4E oscillation axis must be 0 or 1."
        )

    (
        source_array,
        surrogate_array,
    ) = _as_finite_shape_matched_arrays(
        source,
        surrogate,
    )

    if source_array.shape[axis] < 3:
        raise ValueError(
            "At least three grid points are required "
            "for local-turning-point analysis."
        )

    source_turns = _strict_turning_points(
        source_array,
        axis=axis,
    )

    surrogate_turns = _strict_turning_points(
        surrogate_array,
        axis=axis,
    )

    spurious = (
        surrogate_turns
        & ~source_turns
    )

    return OscillationCheckResult(
        axis=axis,
        source_turning_point_count=int(
            np.count_nonzero(
                source_turns
            )
        ),
        surrogate_turning_point_count=int(
            np.count_nonzero(
                surrogate_turns
            )
        ),
        spurious_turning_point_count=int(
            np.count_nonzero(
                spurious
            )
        ),
        spurious_turning_point_locations=(
            _interior_grid_locations(
                spurious,
                axis=axis,
            )
        ),
    )
