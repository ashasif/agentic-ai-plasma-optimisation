"""Phase 5C healthy-reference anomaly baselines.

The implementations in this module are governed by the frozen Phase 5
protocol. Both baselines are fitted exclusively from TRAIN rows labelled
fault_effect_active == False.

No Phase 5 TEST access is implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
from sklearn.ensemble import IsolationForest

from plasma_ai.monitoring.data import (
    DEFAULT_PROTOCOL_PATH,
    MODEL_FEATURES,
)


RESIDUAL_FEATURES = (
    "absorbed_power_relative_residual",
    "flow_relative_residual",
    "pressure_relative_residual",
)

RESIDUAL_FEATURE_INDICES = tuple(
    MODEL_FEATURES.index(name)
    for name in RESIDUAL_FEATURES
)

ROBUST_MAD_FACTOR = 1.4826


def _validate_matrix(
    X: np.ndarray,
    *,
    expected_width: int = len(MODEL_FEATURES),
) -> np.ndarray:
    matrix = np.asarray(
        X,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Phase 5 anomaly feature matrix must be two-dimensional."
        )

    if matrix.shape[1] != expected_width:
        raise ValueError(
            "Phase 5 anomaly feature matrix has wrong feature width."
        )

    if matrix.shape[0] == 0:
        raise ValueError(
            "Phase 5 anomaly feature matrix must contain rows."
        )

    if not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError(
            "Phase 5 anomaly feature matrix contains non-finite values."
        )

    return matrix


def _validate_binary_target(
    y: np.ndarray,
    *,
    rows: int,
) -> np.ndarray:
    values = np.asarray(y)

    if values.ndim != 1:
        raise ValueError(
            "Phase 5 binary target must be one-dimensional."
        )

    if values.shape[0] != rows:
        raise ValueError(
            "Phase 5 anomaly feature/target row-count mismatch."
        )

    if values.dtype != np.bool_:
        if not np.all(
            np.isin(
                values,
                [False, True],
            )
        ):
            raise ValueError(
                "Phase 5 binary target must contain only boolean values."
            )

        values = values.astype(
            np.bool_,
            copy=False,
        )

    return values


def _healthy_training_matrix(
    X: np.ndarray,
    binary_y: np.ndarray,
) -> np.ndarray:
    matrix = _validate_matrix(X)

    target = _validate_binary_target(
        binary_y,
        rows=matrix.shape[0],
    )

    healthy = matrix[
        ~target
    ]

    if healthy.shape[0] == 0:
        raise ValueError(
            "No healthy TRAIN rows are available for anomaly fitting."
        )

    return healthy


def load_phase5_anomaly_contract(
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
) -> dict[str, Any]:
    """Load and verify the frozen Phase 5C anomaly contract."""

    path = Path(
        protocol_path
    )

    protocol = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if protocol.get(
        "protocol_state"
    ) != "FROZEN":
        raise ValueError(
            "Phase 5 protocol must be FROZEN."
        )

    anomaly = protocol.get(
        "anomaly_baselines"
    )

    if not isinstance(
        anomaly,
        dict,
    ):
        raise ValueError(
            "Frozen Phase 5 anomaly-baseline contract is missing."
        )

    robust = anomaly.get(
        "robust_residual"
    )

    isolation = anomaly.get(
        "isolation_forest"
    )

    expected_robust = {
        "training_rows":
            "train rows with fault_effect_active == false",
        "residual_features":
            list(RESIDUAL_FEATURES),
        "robust_scale":
            "1.4826 * MAD; fallback to train-healthy standard deviation if MAD is zero",
        "row_score":
            "max absolute robust z-score",
        "threshold_quantile":
            0.99,
    }

    if robust != expected_robust:
        raise ValueError(
            "Frozen robust-residual anomaly contract drift detected."
        )

    expected_isolation = {
        "training_rows":
            "train rows with fault_effect_active == false",
        "n_estimators":
            500,
        "contamination":
            0.01,
        "random_state":
            20260924,
        "n_jobs":
            1,
    }

    if isolation != expected_isolation:
        raise ValueError(
            "Frozen IsolationForest anomaly contract drift detected."
        )

    return anomaly


@dataclass(frozen=True)
class RobustResidualBaseline:
    """Fitted robust residual anomaly baseline."""

    medians: np.ndarray
    scales: np.ndarray
    scale_sources: tuple[str, ...]
    threshold: float
    threshold_quantile: float
    healthy_fit_rows: int

    def score(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Return max-absolute robust-z anomaly scores."""

        matrix = _validate_matrix(X)

        residuals = matrix[
            :,
            RESIDUAL_FEATURE_INDICES,
        ]

        z = (
            residuals
            - self.medians
        ) / self.scales

        scores = np.max(
            np.abs(z),
            axis=1,
        )

        if not np.all(
            np.isfinite(scores)
        ):
            raise ValueError(
                "Robust anomaly scoring produced non-finite values."
            )

        return scores

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Return True for rows exceeding the frozen threshold."""

        return self.score(X) > self.threshold


@dataclass(frozen=True)
class IsolationForestBaseline:
    """Fitted frozen-configuration IsolationForest baseline."""

    model: IsolationForest
    healthy_fit_rows: int

    def score(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Return anomaly score where larger values are more anomalous.

        sklearn's decision_function is positive for inliers and negative for
        outliers. Negating it gives the Phase 5 convention that larger values
        indicate stronger anomaly evidence.
        """

        matrix = _validate_matrix(X)

        scores = -self.model.decision_function(
            matrix
        )

        if not np.all(
            np.isfinite(scores)
        ):
            raise ValueError(
                "IsolationForest anomaly scoring produced non-finite values."
            )

        return np.asarray(
            scores,
            dtype=np.float64,
        )

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Return True for sklearn IsolationForest outliers."""

        matrix = _validate_matrix(X)

        labels = self.model.predict(
            matrix
        )

        return labels == -1


@dataclass(frozen=True)
class Phase5AnomalyBaselines:
    """Both frozen Phase 5C healthy-reference anomaly baselines."""

    robust_residual: RobustResidualBaseline
    isolation_forest: IsolationForestBaseline


def fit_robust_residual_baseline(
    train_X: np.ndarray,
    train_binary_y: np.ndarray,
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
) -> RobustResidualBaseline:
    """Fit the frozen robust residual baseline on TRAIN-healthy rows."""

    contract = load_phase5_anomaly_contract(
        protocol_path
    )

    healthy = _healthy_training_matrix(
        train_X,
        train_binary_y,
    )

    residuals = healthy[
        :,
        RESIDUAL_FEATURE_INDICES,
    ]

    medians = np.median(
        residuals,
        axis=0,
    )

    absolute_deviations = np.abs(
        residuals
        - medians
    )

    mad = np.median(
        absolute_deviations,
        axis=0,
    )

    std = np.std(
        residuals,
        axis=0,
        ddof=0,
    )

    scales = np.empty(
        len(RESIDUAL_FEATURES),
        dtype=np.float64,
    )

    scale_sources: list[str] = []

    for index in range(
        len(RESIDUAL_FEATURES)
    ):
        robust_scale = (
            ROBUST_MAD_FACTOR
            * mad[index]
        )

        if robust_scale > 0.0:
            scales[index] = robust_scale
            scale_sources.append(
                "mad"
            )
            continue

        fallback = std[index]

        if fallback <= 0.0:
            raise ValueError(
                "Residual feature has zero MAD and zero standard "
                "deviation in TRAIN-healthy rows; frozen fallback "
                "cannot produce a positive scale."
            )

        scales[index] = fallback
        scale_sources.append(
            "standard_deviation_fallback"
        )

    if not np.all(
        np.isfinite(medians)
    ):
        raise ValueError(
            "Robust residual medians are non-finite."
        )

    if not np.all(
        np.isfinite(scales)
    ) or np.any(
        scales <= 0.0
    ):
        raise ValueError(
            "Robust residual scales must be finite and positive."
        )

    z = (
        residuals
        - medians
    ) / scales

    healthy_scores = np.max(
        np.abs(z),
        axis=1,
    )

    threshold_quantile = float(
        contract[
            "robust_residual"
        ][
            "threshold_quantile"
        ]
    )

    threshold = float(
        np.quantile(
            healthy_scores,
            threshold_quantile,
            method="linear",
        )
    )

    medians = np.asarray(
        medians,
        dtype=np.float64,
    )
    scales = np.asarray(
        scales,
        dtype=np.float64,
    )

    medians.setflags(
        write=False
    )
    scales.setflags(
        write=False
    )

    return RobustResidualBaseline(
        medians=medians,
        scales=scales,
        scale_sources=tuple(
            scale_sources
        ),
        threshold=threshold,
        threshold_quantile=threshold_quantile,
        healthy_fit_rows=int(
            healthy.shape[0]
        ),
    )


def fit_isolation_forest_baseline(
    train_X: np.ndarray,
    train_binary_y: np.ndarray,
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
) -> IsolationForestBaseline:
    """Fit frozen IsolationForest configuration on TRAIN-healthy rows."""

    contract = load_phase5_anomaly_contract(
        protocol_path
    )

    healthy = _healthy_training_matrix(
        train_X,
        train_binary_y,
    )

    config = contract[
        "isolation_forest"
    ]

    model = IsolationForest(
        n_estimators=int(
            config["n_estimators"]
        ),
        contamination=float(
            config["contamination"]
        ),
        random_state=int(
            config["random_state"]
        ),
        n_jobs=int(
            config["n_jobs"]
        ),
    )

    model.fit(
        healthy
    )

    return IsolationForestBaseline(
        model=model,
        healthy_fit_rows=int(
            healthy.shape[0]
        ),
    )


def fit_phase5_anomaly_baselines(
    train_X: np.ndarray,
    train_binary_y: np.ndarray,
    *,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
) -> Phase5AnomalyBaselines:
    """Fit both frozen Phase 5C baselines on TRAIN-healthy rows."""

    return Phase5AnomalyBaselines(
        robust_residual=fit_robust_residual_baseline(
            train_X,
            train_binary_y,
            protocol_path=protocol_path,
        ),
        isolation_forest=fit_isolation_forest_baseline(
            train_X,
            train_binary_y,
            protocol_path=protocol_path,
        ),
    )
