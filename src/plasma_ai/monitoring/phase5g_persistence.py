"""Trusted persistence and reproducible inference for Phase 5G.

This module wraps the already-frozen Phase 5 detector and diagnoser
artifacts.  It never retrains, reconstructs, copies, or reserializes
those models.

The trusted loading path verifies exact artifact hashes before pickle
deserialization.  Runtime inference accepts only the six frozen raw
monitoring inputs and deterministically constructs the frozen nine-feature
model representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import pickle
import platform
import subprocess

import numpy as np
import scipy
import sklearn
from sklearn.ensemble import ExtraTreesClassifier

from plasma_ai.monitoring.data import (
    MODEL_FEATURES,
    RAW_FEATURES,
)
from plasma_ai.monitoring.redevelopment import (
    FittedRedevelopmentDiagnoserCandidate,
)
from plasma_ai.monitoring.supervised import (
    FittedDetectorCandidate,
)


DEFAULT_PHASE5G_PROTOCOL_PATH = Path(
    "configs/phase5/phase5g_protocol.json"
)

DEFAULT_MANIFEST_PATH = Path(
    "artifacts/phase5/monitoring_manifest.json"
)

EXPECTED_PHASE5G_PROTOCOL_SHA256 = (
    "2ca15b486bf15c3c439e19607812358f"
    "632151c94de15541ae6d27e04fe1574c"
)

MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Phase5GPredictions:
    """Deterministic Phase 5G monitoring predictions."""

    active_probability: np.ndarray
    fault_active: np.ndarray
    diagnostic_state: np.ndarray


@dataclass(frozen=True)
class LoadedPhase5GMonitoring:
    """Trusted loaded Phase 5G detector/diagnoser pair."""

    detector: FittedDetectorCandidate
    diagnoser: FittedRedevelopmentDiagnoserCandidate
    manifest: dict[str, Any]

    def predict(
        self,
        raw_values: Any,
    ) -> Phase5GPredictions:
        """Predict frozen detector and end-to-end diagnostic state."""

        features = build_phase5g_model_features(
            raw_values
        )

        probabilities = np.asarray(
            self.detector.predict_active_probability(
                features
            ),
            dtype=np.float64,
        )

        decisions = np.asarray(
            self.detector.predict(
                features
            ),
            dtype=bool,
        )

        expected_shape = (
            features.shape[0],
        )

        if probabilities.shape != expected_shape:
            raise RuntimeError(
                "Detector active-probability output has "
                "an unexpected shape."
            )

        if decisions.shape != expected_shape:
            raise RuntimeError(
                "Detector decision output has an unexpected shape."
            )

        if not np.all(
            np.isfinite(
                probabilities
            )
        ):
            raise RuntimeError(
                "Detector produced non-finite active probabilities."
            )

        threshold_decisions = (
            probabilities
            >= float(
                self.detector.threshold
            )
        )

        if not np.array_equal(
            decisions,
            threshold_decisions,
        ):
            raise RuntimeError(
                "Detector decision output does not match the "
                "frozen threshold rule."
            )

        prediction_contract = self.manifest[
            "prediction_contract"
        ]

        inactive_state = str(
            prediction_contract[
                "inactive_state"
            ]
        )

        allowed_states = tuple(
            str(value)
            for value in prediction_contract[
                "end_to_end_states"
            ]
        )

        max_state_length = max(
            len(value)
            for value in allowed_states
        )

        states = np.full(
            features.shape[0],
            inactive_state,
            dtype=f"<U{max_state_length}",
        )

        active_indices = np.flatnonzero(
            decisions
        )

        if active_indices.size:
            active_predictions = np.asarray(
                self.diagnoser.predict(
                    features[
                        active_indices
                    ]
                ),
                dtype=str,
            )

            if active_predictions.shape != (
                active_indices.size,
            ):
                raise RuntimeError(
                    "Diagnoser prediction output has "
                    "an unexpected shape."
                )

            unknown = set(
                active_predictions.tolist()
            ) - set(
                allowed_states
            )

            if unknown:
                raise RuntimeError(
                    "Diagnoser produced states outside the "
                    "frozen Phase 5G runtime contract: "
                    f"{sorted(unknown)!r}"
                )

            if inactive_state in set(
                active_predictions.tolist()
            ):
                raise RuntimeError(
                    "Diagnoser produced the inactive state for "
                    "a detector-positive row."
                )

            states[
                active_indices
            ] = active_predictions

        probabilities = np.array(
            probabilities,
            dtype=np.float64,
            copy=True,
        )

        decisions = np.array(
            decisions,
            dtype=bool,
            copy=True,
        )

        states = np.array(
            states,
            copy=True,
        )

        probabilities.setflags(
            write=False
        )

        decisions.setflags(
            write=False
        )

        states.setflags(
            write=False
        )

        return Phase5GPredictions(
            active_probability=probabilities,
            fault_active=decisions,
            diagnostic_state=states,
        )


def _sha256_bytes(
    payload: bytes,
) -> str:
    """Return the SHA-256 digest for bytes."""

    return hashlib.sha256(
        payload
    ).hexdigest()


def _load_json(
    path: str | Path,
) -> dict[str, Any]:
    """Load one JSON object."""

    source = Path(
        path
    )

    payload = json.loads(
        source.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"Expected JSON object at {source}."
        )

    return payload


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


def _runtime_metadata() -> dict[str, str]:
    """Return scientific-runtime provenance."""

    return {
        "python_version":
            platform.python_version(),

        "python_implementation":
            platform.python_implementation(),

        "platform":
            platform.platform(),

        "numpy_version":
            np.__version__,

        "scipy_version":
            scipy.__version__,

        "scikit_learn_version":
            sklearn.__version__,
    }


def _git_state() -> tuple[str, bool]:
    """Return current commit and clean-state flag."""

    commit = subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()

    status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        text=True,
    )

    return (
        commit,
        not bool(
            status.strip()
        ),
    )


def load_phase5g_protocol(
    path: str | Path = DEFAULT_PHASE5G_PROTOCOL_PATH,
) -> dict[str, Any]:
    """Load and verify the frozen Phase 5G protocol."""

    source = Path(
        path
    )

    payload = source.read_bytes()

    observed_hash = _sha256_bytes(
        payload
    )

    if observed_hash != EXPECTED_PHASE5G_PROTOCOL_SHA256:
        raise ValueError(
            "Phase 5G protocol SHA-256 mismatch: "
            f"expected {EXPECTED_PHASE5G_PROTOCOL_SHA256}, "
            f"observed {observed_hash}."
        )

    protocol = json.loads(
        payload.decode(
            "utf-8"
        )
    )

    if protocol.get(
        "phase"
    ) != "5G":
        raise ValueError(
            "Unexpected Phase 5G protocol phase."
        )

    if protocol.get(
        "protocol_state"
    ) != "FROZEN":
        raise ValueError(
            "Phase 5G protocol is not frozen."
        )

    if protocol[
        "governance"
    ][
        "phase5_test_dataset_access_permitted"
    ] is not False:
        raise ValueError(
            "Frozen Phase 5G protocol unexpectedly permits TEST access."
        )

    if protocol[
        "governance"
    ][
        "frozen_models_must_remain_byte_identical"
    ] is not True:
        raise ValueError(
            "Frozen Phase 5G protocol does not require byte-identical models."
        )

    return protocol


def _validate_source_evidence(
    protocol: dict[str, Any],
) -> None:
    """Verify frozen source evidence without reopening the TEST dataset."""

    for record in protocol[
        "source_evidence"
    ].values():
        path = Path(
            record[
                "path"
            ]
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Frozen Phase 5G source evidence not found: {path}"
            )

        observed = _sha256_bytes(
            path.read_bytes()
        )

        expected = str(
            record[
                "sha256"
            ]
        )

        if observed != expected:
            raise ValueError(
                "Frozen Phase 5G source-evidence SHA-256 mismatch for "
                f"{path}: expected {expected}, observed {observed}."
            )

    for key in (
        "detector",
        "diagnoser",
    ):
        record = protocol[
            "model_artifacts"
        ][
            key
        ]

        path = Path(
            record[
                "repository_path"
            ]
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Frozen Phase 5 model artifact not found: {path}"
            )

        observed = _sha256_bytes(
            path.read_bytes()
        )

        expected = str(
            record[
                "sha256"
            ]
        )

        if observed != expected:
            raise ValueError(
                "Frozen Phase 5 model SHA-256 mismatch for "
                f"{path}: expected {expected}, observed {observed}."
            )


def _validated_raw_inference_matrix(
    values: Any,
) -> np.ndarray:
    """Validate public six-feature Phase 5G runtime input."""

    try:
        matrix = np.asarray(
            values,
            dtype=np.float64,
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Phase 5G runtime input must be convertible to float64."
        ) from exc

    expected_width = len(
        RAW_FEATURES
    )

    if matrix.ndim == 1:
        if matrix.shape != (
            expected_width,
        ):
            raise ValueError(
                "One-dimensional Phase 5G runtime input must "
                f"have shape ({expected_width},)."
            )

        matrix = matrix.reshape(
            1,
            expected_width,
        )

    elif matrix.ndim == 2:
        if matrix.shape[1] != expected_width:
            raise ValueError(
                "Two-dimensional Phase 5G runtime input must "
                f"have shape (n_rows, {expected_width})."
            )

        if matrix.shape[0] == 0:
            raise ValueError(
                "Phase 5G runtime input must contain at least one row."
            )

    else:
        raise ValueError(
            "Phase 5G runtime input must be rank 1 or rank 2."
        )

    if not np.all(
        np.isfinite(
            matrix
        )
    ):
        raise ValueError(
            "Phase 5G runtime input contains non-finite values."
        )

    nominal_power = matrix[
        :,
        0,
    ]

    target_pressure = matrix[
        :,
        1,
    ]

    nominal_flow = matrix[
        :,
        2,
    ]

    if np.any(
        nominal_power <= 0.0
    ):
        raise ValueError(
            "nominal_absorbed_power_W must be positive."
        )

    if np.any(
        target_pressure <= 0.0
    ):
        raise ValueError(
            "target_pressure_mTorr must be positive."
        )

    if np.any(
        nominal_flow <= 0.0
    ):
        raise ValueError(
            "nominal_flow_sccm must be positive."
        )

    result = np.array(
        matrix,
        dtype=np.float64,
        copy=True,
    )

    result.setflags(
        write=False
    )

    return result


def build_phase5g_model_features(
    raw_values: Any,
) -> np.ndarray:
    """Construct the frozen nine-feature model matrix."""

    raw = _validated_raw_inference_matrix(
        raw_values
    )

    absorbed_power_relative_residual = (
        raw[
            :,
            3,
        ]
        / raw[
            :,
            0,
        ]
        - 1.0
    )

    flow_relative_residual = (
        raw[
            :,
            4,
        ]
        / raw[
            :,
            2,
        ]
        - 1.0
    )

    pressure_relative_residual = (
        raw[
            :,
            5,
        ]
        / raw[
            :,
            1,
        ]
        - 1.0
    )

    matrix = np.column_stack(
        (
            raw,
            absorbed_power_relative_residual,
            flow_relative_residual,
            pressure_relative_residual,
        )
    ).astype(
        np.float64,
        copy=False,
    )

    if matrix.shape[1] != len(
        MODEL_FEATURES
    ):
        raise RuntimeError(
            "Constructed Phase 5G model matrix has an unexpected width."
        )

    if not np.all(
        np.isfinite(
            matrix
        )
    ):
        raise ValueError(
            "Constructed Phase 5G model features contain non-finite values."
        )

    result = np.array(
        matrix,
        dtype=np.float64,
        copy=True,
    )

    result.setflags(
        write=False
    )

    return result


def _read_verified_pickle_bytes(
    path: Path,
    *,
    expected_sha256: str,
) -> bytes:
    """Verify trusted artifact bytes before pickle deserialization."""

    if not path.exists():
        raise FileNotFoundError(
            f"Persisted Phase 5 model artifact not found: {path}"
        )

    payload = path.read_bytes()

    observed = _sha256_bytes(
        payload
    )

    if observed != expected_sha256:
        raise ValueError(
            "Persisted Phase 5 model artifact SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {observed}."
        )

    return payload


def _validate_estimator_parameters(
    model: ExtraTreesClassifier,
    expected: dict[str, Any],
    *,
    model_label: str,
) -> None:
    """Validate frozen ExtraTrees parameters."""

    observed = model.get_params(
        deep=False
    )

    for name, expected_value in expected.items():
        if observed.get(
            name
        ) != expected_value:
            raise ValueError(
                f"Persisted {model_label} estimator parameter "
                f"{name!r} does not match the frozen Phase 5G contract."
            )


def _wrapper_type_name(
    value: Any,
) -> str:
    """Return fully-qualified wrapper class name."""

    cls = type(
        value
    )

    return (
        cls.__module__
        + "."
        + cls.__qualname__
    )


def _validate_loaded_wrappers(
    detector: Any,
    diagnoser: Any,
    protocol: dict[str, Any],
) -> None:
    """Validate exact frozen wrapper and estimator contracts."""

    detector_expected = protocol[
        "model_artifacts"
    ][
        "detector"
    ]

    diagnoser_expected = protocol[
        "model_artifacts"
    ][
        "diagnoser"
    ]

    if not isinstance(
        detector,
        FittedDetectorCandidate,
    ):
        raise ValueError(
            "Persisted detector has an unexpected wrapper type."
        )

    if _wrapper_type_name(
        detector
    ) != detector_expected[
        "wrapper_type"
    ]:
        raise ValueError(
            "Persisted detector wrapper type does not match protocol."
        )

    if detector.candidate_id != detector_expected[
        "candidate_id"
    ]:
        raise ValueError(
            "Persisted detector candidate ID does not match protocol."
        )

    if int(
        detector.fit_rows
    ) != int(
        detector_expected[
            "fit_rows"
        ]
    ):
        raise ValueError(
            "Persisted detector fit-row count does not match protocol."
        )

    if float(
        detector.threshold
    ) != float(
        detector_expected[
            "threshold"
        ]
    ):
        raise ValueError(
            "Persisted detector threshold does not match protocol."
        )

    if not isinstance(
        detector.model,
        ExtraTreesClassifier,
    ):
        raise ValueError(
            "Persisted detector estimator has an unexpected type."
        )

    _validate_estimator_parameters(
        detector.model,
        detector_expected[
            "parameters"
        ],
        model_label="detector",
    )

    if not isinstance(
        diagnoser,
        FittedRedevelopmentDiagnoserCandidate,
    ):
        raise ValueError(
            "Persisted diagnoser has an unexpected wrapper type."
        )

    if _wrapper_type_name(
        diagnoser
    ) != diagnoser_expected[
        "wrapper_type"
    ]:
        raise ValueError(
            "Persisted diagnoser wrapper type does not match protocol."
        )

    if diagnoser.candidate_id != diagnoser_expected[
        "candidate_id"
    ]:
        raise ValueError(
            "Persisted diagnoser candidate ID does not match protocol."
        )

    if int(
        diagnoser.fit_rows
    ) != int(
        diagnoser_expected[
            "fit_rows"
        ]
    ):
        raise ValueError(
            "Persisted diagnoser fit-row count does not match protocol."
        )

    if (
        diagnoser.train_majority_class
        != diagnoser_expected[
            "train_majority_class"
        ]
    ):
        raise ValueError(
            "Persisted diagnoser majority class does not match protocol."
        )

    if not isinstance(
        diagnoser.model,
        ExtraTreesClassifier,
    ):
        raise ValueError(
            "Persisted diagnoser estimator has an unexpected type."
        )

    _validate_estimator_parameters(
        diagnoser.model,
        diagnoser_expected[
            "parameters"
        ],
        model_label="diagnoser",
    )

    observed_classes = tuple(
        str(value)
        for value in diagnoser.model.classes_
    )

    expected_classes = tuple(
        str(value)
        for value in diagnoser_expected[
            "classes"
        ]
    )

    if observed_classes != expected_classes:
        raise ValueError(
            "Persisted diagnoser classes do not match protocol."
        )


def _load_verified_frozen_models(
    protocol: dict[str, Any],
) -> tuple[
    FittedDetectorCandidate,
    FittedRedevelopmentDiagnoserCandidate,
]:
    """Load exact frozen model artifacts after SHA-256 verification."""

    detector_record = protocol[
        "model_artifacts"
    ][
        "detector"
    ]

    diagnoser_record = protocol[
        "model_artifacts"
    ][
        "diagnoser"
    ]

    detector_bytes = _read_verified_pickle_bytes(
        Path(
            detector_record[
                "repository_path"
            ]
        ),
        expected_sha256=str(
            detector_record[
                "sha256"
            ]
        ),
    )

    diagnoser_bytes = _read_verified_pickle_bytes(
        Path(
            diagnoser_record[
                "repository_path"
            ]
        ),
        expected_sha256=str(
            diagnoser_record[
                "sha256"
            ]
        ),
    )

    # These are trusted project artifacts whose exact frozen bytes
    # have been verified immediately above. Never use this mechanism
    # for arbitrary or untrusted pickle files.
    detector = pickle.loads(
        detector_bytes
    )

    diagnoser = pickle.loads(
        diagnoser_bytes
    )

    _validate_loaded_wrappers(
        detector,
        diagnoser,
        protocol,
    )

    return (
        detector,
        diagnoser,
    )


def _build_manifest(
    protocol: dict[str, Any],
    *,
    source_git_commit: str,
    repository_clean_state: bool,
    artifact_creation_timestamp_utc: str,
) -> dict[str, Any]:
    """Build the Phase 5G runtime manifest."""

    runtime = _runtime_metadata()

    return {
        "phase":
            "5G",

        "stage":
            "trusted_monitoring_inference_manifest",

        "schema_version":
            MANIFEST_SCHEMA_VERSION,

        "status":
            "ready",

        "feature_contract": {
            "raw_feature_order":
                list(
                    protocol[
                        "runtime_input_contract"
                    ][
                        "raw_feature_order"
                    ]
                ),

            "model_feature_order":
                list(
                    protocol[
                        "feature_construction"
                    ][
                        "model_feature_order"
                    ]
                ),

            "strictly_positive_inputs":
                list(
                    protocol[
                        "runtime_input_contract"
                    ][
                        "strictly_positive_inputs"
                    ]
                ),

            "derived_features":
                dict(
                    protocol[
                        "feature_construction"
                    ][
                        "derived_features"
                    ]
                ),
        },

        "prediction_contract": {
            "detector_threshold":
                protocol[
                    "runtime_prediction_contract"
                ][
                    "detector_threshold"
                ],

            "diagnoser_execution":
                protocol[
                    "runtime_prediction_contract"
                ][
                    "diagnoser_execution"
                ],

            "end_to_end_states":
                list(
                    protocol[
                        "runtime_prediction_contract"
                    ][
                        "end_to_end_states"
                    ]
                ),

            "inactive_state":
                protocol[
                    "runtime_prediction_contract"
                ][
                    "inactive_state"
                ],

            "prediction_output_fields":
                list(
                    protocol[
                        "runtime_prediction_contract"
                    ][
                        "prediction_output_fields"
                    ]
                ),
        },

        "models": {
            "detector":
                json.loads(
                    json.dumps(
                        protocol[
                            "model_artifacts"
                        ][
                            "detector"
                        ]
                    )
                ),

            "diagnoser":
                json.loads(
                    json.dumps(
                        protocol[
                            "model_artifacts"
                        ][
                            "diagnoser"
                        ]
                    )
                ),
        },

        "provenance": {
            "phase5g_protocol": {
                "path":
                    str(
                        DEFAULT_PHASE5G_PROTOCOL_PATH.as_posix()
                    ),

                "sha256":
                    EXPECTED_PHASE5G_PROTOCOL_SHA256,
            },

            "source_git_commit":
                source_git_commit,

            "repository_clean_state":
                repository_clean_state,

            "artifact_creation_timestamp_utc":
                artifact_creation_timestamp_utc,

            "python_version":
                runtime[
                    "python_version"
                ],

            "python_implementation":
                runtime[
                    "python_implementation"
                ],

            "platform":
                runtime[
                    "platform"
                ],

            "numpy_version":
                runtime[
                    "numpy_version"
                ],

            "scipy_version":
                runtime[
                    "scipy_version"
                ],

            "scikit_learn_version":
                runtime[
                    "scikit_learn_version"
                ],

            "source_evidence":
                json.loads(
                    json.dumps(
                        protocol[
                            "source_evidence"
                        ]
                    )
                ),
        },

        "scientific_scope":
            json.loads(
                json.dumps(
                    protocol[
                        "scientific_scope"
                    ]
                )
            ),
    }


def _validate_manifest_against_protocol(
    manifest: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    """Validate persisted manifest against the frozen protocol."""

    if manifest.get(
        "phase"
    ) != "5G":
        raise ValueError(
            "Unexpected Phase 5G manifest phase."
        )

    if manifest.get(
        "stage"
    ) != "trusted_monitoring_inference_manifest":
        raise ValueError(
            "Unexpected Phase 5G manifest stage."
        )

    if manifest.get(
        "schema_version"
    ) != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported Phase 5G manifest schema version."
        )

    if manifest.get(
        "status"
    ) != "ready":
        raise ValueError(
            "Phase 5G manifest is not ready."
        )

    expected_feature_contract = {
        "raw_feature_order":
            list(
                protocol[
                    "runtime_input_contract"
                ][
                    "raw_feature_order"
                ]
            ),

        "model_feature_order":
            list(
                protocol[
                    "feature_construction"
                ][
                    "model_feature_order"
                ]
            ),

        "strictly_positive_inputs":
            list(
                protocol[
                    "runtime_input_contract"
                ][
                    "strictly_positive_inputs"
                ]
            ),

        "derived_features":
            dict(
                protocol[
                    "feature_construction"
                ][
                    "derived_features"
                ]
            ),
    }

    if manifest.get(
        "feature_contract"
    ) != expected_feature_contract:
        raise ValueError(
            "Phase 5G manifest feature contract does not match protocol."
        )

    expected_prediction_contract = {
        "detector_threshold":
            protocol[
                "runtime_prediction_contract"
            ][
                "detector_threshold"
            ],

        "diagnoser_execution":
            protocol[
                "runtime_prediction_contract"
            ][
                "diagnoser_execution"
            ],

        "end_to_end_states":
            list(
                protocol[
                    "runtime_prediction_contract"
                ][
                    "end_to_end_states"
                ]
            ),

        "inactive_state":
            protocol[
                "runtime_prediction_contract"
            ][
                "inactive_state"
            ],

        "prediction_output_fields":
            list(
                protocol[
                    "runtime_prediction_contract"
                ][
                    "prediction_output_fields"
                ]
            ),
    }

    if manifest.get(
        "prediction_contract"
    ) != expected_prediction_contract:
        raise ValueError(
            "Phase 5G manifest prediction contract does not match protocol."
        )

    for key in (
        "detector",
        "diagnoser",
    ):
        expected_record = protocol[
            "model_artifacts"
        ][
            key
        ]

        observed_record = manifest.get(
            "models",
            {},
        ).get(
            key
        )

        if observed_record != expected_record:
            raise ValueError(
                f"Phase 5G manifest {key} metadata does not match protocol."
            )

    provenance = manifest.get(
        "provenance",
        {},
    )

    expected_protocol_record = {
        "path":
            str(
                DEFAULT_PHASE5G_PROTOCOL_PATH.as_posix()
            ),

        "sha256":
            EXPECTED_PHASE5G_PROTOCOL_SHA256,
    }

    if provenance.get(
        "phase5g_protocol"
    ) != expected_protocol_record:
        raise ValueError(
            "Phase 5G manifest protocol provenance does not match."
        )

    if provenance.get(
        "source_evidence"
    ) != protocol[
        "source_evidence"
    ]:
        raise ValueError(
            "Phase 5G manifest source evidence does not match protocol."
        )

    if provenance.get(
        "repository_clean_state"
    ) is not True:
        raise ValueError(
            "Phase 5G manifest was not created from a clean repository."
        )

    source_commit = provenance.get(
        "source_git_commit"
    )

    if not isinstance(
        source_commit,
        str,
    ) or not source_commit:
        raise ValueError(
            "Phase 5G manifest source Git commit is invalid."
        )

    timestamp = provenance.get(
        "artifact_creation_timestamp_utc"
    )

    if not isinstance(
        timestamp,
        str,
    ) or not timestamp:
        raise ValueError(
            "Phase 5G manifest creation timestamp is invalid."
        )

    if manifest.get(
        "scientific_scope"
    ) != protocol[
        "scientific_scope"
    ]:
        raise ValueError(
            "Phase 5G manifest scientific scope does not match protocol."
        )


def _validate_runtime_compatibility(
    manifest: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    """Require exact frozen scientific-runtime versions."""

    current = _runtime_metadata()

    required = protocol[
        "runtime_compatibility"
    ][
        "required_exact_versions"
    ]

    mapping = {
        "python":
            "python_version",

        "numpy":
            "numpy_version",

        "scipy":
            "scipy_version",

        "scikit_learn":
            "scikit_learn_version",
    }

    provenance = manifest[
        "provenance"
    ]

    for protocol_name, runtime_name in mapping.items():
        required_value = str(
            required[
                protocol_name
            ]
        )

        current_value = str(
            current[
                runtime_name
            ]
        )

        persisted_value = str(
            provenance.get(
                runtime_name
            )
        )

        if current_value != required_value:
            raise RuntimeError(
                "Current Phase 5G scientific runtime mismatch for "
                f"{protocol_name}: expected {required_value!r}, "
                f"observed {current_value!r}."
            )

        if persisted_value != required_value:
            raise RuntimeError(
                "Persisted Phase 5G scientific runtime mismatch for "
                f"{protocol_name}: expected {required_value!r}, "
                f"persisted {persisted_value!r}."
            )


def write_phase5g_manifest(
    output_path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    protocol_path: str | Path = DEFAULT_PHASE5G_PROTOCOL_PATH,
) -> Path:
    """Create the manifest around exact existing frozen model bytes."""

    protocol = load_phase5g_protocol(
        protocol_path
    )

    _validate_source_evidence(
        protocol
    )

    destination = Path(
        output_path
    )

    if destination.exists():
        raise FileExistsError(
            "Phase 5G manifest persistence refuses to overwrite "
            f"existing artifact: {destination}"
        )

    git_commit, repository_clean = _git_state()

    if not repository_clean:
        raise RuntimeError(
            "Phase 5G production manifest may only be generated "
            "from a clean committed repository state."
        )

    # Trusted load validates exact model bytes, wrappers, estimator
    # types, parameters, candidate identities, and diagnosis classes.
    _load_verified_frozen_models(
        protocol
    )

    manifest = _build_manifest(
        protocol,
        source_git_commit=git_commit,
        repository_clean_state=repository_clean,
        artifact_creation_timestamp_utc=_utc_now(),
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return destination


def load_phase5g_monitoring(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    protocol_path: str | Path = DEFAULT_PHASE5G_PROTOCOL_PATH,
) -> LoadedPhase5GMonitoring:
    """Load trusted frozen Phase 5G monitoring inference."""

    protocol = load_phase5g_protocol(
        protocol_path
    )

    manifest = _load_json(
        manifest_path
    )

    _validate_manifest_against_protocol(
        manifest,
        protocol,
    )

    _validate_runtime_compatibility(
        manifest,
        protocol,
    )

    detector_record = manifest[
        "models"
    ][
        "detector"
    ]

    diagnoser_record = manifest[
        "models"
    ][
        "diagnoser"
    ]

    detector_bytes = _read_verified_pickle_bytes(
        Path(
            detector_record[
                "repository_path"
            ]
        ),
        expected_sha256=str(
            detector_record[
                "sha256"
            ]
        ),
    )

    diagnoser_bytes = _read_verified_pickle_bytes(
        Path(
            diagnoser_record[
                "repository_path"
            ]
        ),
        expected_sha256=str(
            diagnoser_record[
                "sha256"
            ]
        ),
    )

    # Exact trusted project bytes were verified before deserialization.
    detector = pickle.loads(
        detector_bytes
    )

    diagnoser = pickle.loads(
        diagnoser_bytes
    )

    _validate_loaded_wrappers(
        detector,
        diagnoser,
        protocol,
    )

    return LoadedPhase5GMonitoring(
        detector=detector,
        diagnoser=diagnoser,
        manifest=manifest,
    )


def validate_phase5g_preflight(
    *,
    protocol_path: str | Path = DEFAULT_PHASE5G_PROTOCOL_PATH,
) -> dict[str, Any]:
    """Validate Phase 5G frozen inputs without any monitoring dataset."""

    protocol = load_phase5g_protocol(
        protocol_path
    )

    _validate_source_evidence(
        protocol
    )

    runtime = _runtime_metadata()

    required = protocol[
        "runtime_compatibility"
    ][
        "required_exact_versions"
    ]

    runtime_checks = {
        "python":
            runtime[
                "python_version"
            ],

        "numpy":
            runtime[
                "numpy_version"
            ],

        "scipy":
            runtime[
                "scipy_version"
            ],

        "scikit_learn":
            runtime[
                "scikit_learn_version"
            ],
    }

    for key, observed in runtime_checks.items():
        if str(
            observed
        ) != str(
            required[
                key
            ]
        ):
            raise RuntimeError(
                f"Phase 5G preflight runtime mismatch for {key}."
            )

    detector, diagnoser = _load_verified_frozen_models(
        protocol
    )

    return {
        "phase":
            "5G",

        "protocol_sha256":
            EXPECTED_PHASE5G_PROTOCOL_SHA256,

        "detector_candidate_id":
            detector.candidate_id,

        "detector_threshold":
            float(
                detector.threshold
            ),

        "detector_fit_rows":
            int(
                detector.fit_rows
            ),

        "diagnoser_candidate_id":
            diagnoser.candidate_id,

        "diagnoser_fit_rows":
            int(
                diagnoser.fit_rows
            ),

        "runtime":
            runtime,

        "phase5_test_dataset_accessed":
            False,

        "models_retrained":
            False,

        "models_reserialized":
            False,
    }
