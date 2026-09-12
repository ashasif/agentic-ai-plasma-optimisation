"""Phase 4G persisted-surrogate and reproducible-inference support.

This module persists and reloads the already-frozen Phase 4 final surrogate.

It does not perform model selection, retuning, TEST evaluation, or speed
benchmarking.

Production artifact generation is permitted only from a clean committed
repository state. Persisted pickle files are trusted project artifacts only;
their SHA-256 digests are verified before deserialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import pickle
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy
import sklearn
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)

from plasma_ai.surrogate.data import file_sha256
from plasma_ai.surrogate.phase4er_final_refit import (
    EXPECTED_FINAL_REFIT_ROWS,
    EXPECTED_FEATURE_NAMES,
    Phase4ERFinalRefit,
    run_phase4er_final_refit,
)
from plasma_ai.surrogate.transforms import TargetTransform


DEFAULT_PHASE4G_PROTOCOL_PATH = Path(
    "configs/phase4/phase4g_protocol.json"
)

DEFAULT_ARTIFACT_DIRECTORY = Path(
    "artifacts/phase4"
)

DEFAULT_MANIFEST_PATH = (
    DEFAULT_ARTIFACT_DIRECTORY
    / "surrogate_manifest.json"
)

DENSITY_MODEL_FILENAME = "density_model.pkl"
TEMPERATURE_MODEL_FILENAME = "temperature_model.pkl"

MANIFEST_SCHEMA_VERSION = "1.0.0"

EXPECTED_PHASE4G_STATUS = (
    "frozen_before_benchmark_execution"
)


@dataclass(frozen=True)
class Phase4GPredictions:
    """Physical-scale predictions from the frozen persisted surrogate."""

    electron_density_m3: NDArray[np.float64]
    electron_temperature_eV: NDArray[np.float64]


@dataclass(frozen=True)
class LoadedPhase4GSurrogate:
    """Validated persisted Phase 4G surrogate."""

    density_model: HistGradientBoostingRegressor
    temperature_model: ExtraTreesRegressor
    density_transform: TargetTransform
    temperature_transform: TargetTransform
    manifest: dict[str, Any]

    def predict_physical(
        self,
        X: ArrayLike,
    ) -> Phase4GPredictions:
        """Predict both frozen targets on their physical scales."""

        matrix = _validated_inference_matrix(
            X
        )

        density_transformed = np.asarray(
            self.density_model.predict(
                matrix
            ),
            dtype=np.float64,
        )

        temperature_transformed = np.asarray(
            self.temperature_model.predict(
                matrix
            ),
            dtype=np.float64,
        )

        expected_shape = (
            matrix.shape[0],
        )

        if density_transformed.shape != expected_shape:
            raise RuntimeError(
                "Density surrogate returned an unexpected "
                "prediction shape."
            )

        if temperature_transformed.shape != expected_shape:
            raise RuntimeError(
                "Temperature surrogate returned an unexpected "
                "prediction shape."
            )

        density = self.density_transform.inverse(
            density_transformed
        )

        temperature = self.temperature_transform.inverse(
            temperature_transformed
        )

        if not (
            np.isfinite(density).all()
            and np.isfinite(temperature).all()
        ):
            raise RuntimeError(
                "Persisted surrogate produced non-finite "
                "physical predictions."
            )

        return Phase4GPredictions(
            electron_density_m3=density,
            electron_temperature_eV=temperature,
        )


def _load_json(
    path: str | Path,
) -> dict:
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def _utc_now() -> str:
    """Return a UTC timestamp suitable for machine-readable provenance."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def _runtime_metadata() -> dict[str, str]:
    """Capture the runtime required by the Phase 4G provenance contract."""

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
    }


def _git_state() -> tuple[str, bool]:
    """Return current Git HEAD and whether the working tree is clean."""

    commit = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    porcelain = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    return (
        commit,
        porcelain.strip() == "",
    )


def load_phase4g_protocol(
    path: str | Path = DEFAULT_PHASE4G_PROTOCOL_PATH,
) -> dict:
    """Load and validate the frozen Phase 4G protocol."""

    payload = _load_json(
        path
    )

    if payload.get("phase") != "4G":
        raise ValueError(
            "Unexpected Phase 4G protocol phase marker."
        )

    if payload.get("status") != EXPECTED_PHASE4G_STATUS:
        raise ValueError(
            "Phase 4G protocol is not frozen before benchmark execution."
        )

    if payload[
        "test_status"
    ].get(
        "rerun_permitted"
    ) is not False:
        raise ValueError(
            "Phase 4G protocol must prohibit TEST reruns."
        )

    persistence = payload[
        "persistence"
    ]

    if persistence.get(
        "serialization"
    ) != "pickle":
        raise ValueError(
            "Unexpected Phase 4G persistence representation."
        )

    if int(
        persistence.get(
            "pickle_protocol",
            -1,
        )
    ) != 5:
        raise ValueError(
            "Phase 4G requires pickle protocol 5."
        )

    if persistence.get(
        "verify_hash_before_deserialization"
    ) is not True:
        raise ValueError(
            "Phase 4G requires artifact hash verification "
            "before deserialization."
        )

    return payload


def _validated_inference_matrix(
    values: ArrayLike,
) -> NDArray[np.float64]:
    """Validate the frozen two-feature inference input contract."""

    matrix = np.asarray(
        values,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Inference features must be a two-dimensional matrix."
        )

    if matrix.shape[0] == 0:
        raise ValueError(
            "Inference features must contain at least one row."
        )

    if matrix.shape[1] != len(
        EXPECTED_FEATURE_NAMES
    ):
        raise ValueError(
            "Phase 4G inference requires exactly two frozen features "
            "in the order nominal_absorbed_power_W, "
            "target_pressure_mTorr."
        )

    if not np.isfinite(
        matrix
    ).all():
        raise ValueError(
            "Inference features must contain only finite values."
        )

    return matrix


def _validate_source_evidence(
    protocol: dict,
) -> None:
    """Verify all frozen Phase 4G source-evidence hashes."""

    for key, evidence in protocol[
        "source_evidence"
    ].items():
        path = Path(
            evidence["path"]
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Frozen Phase 4G source evidence not found: {path}"
            )

        observed = file_sha256(
            path
        )

        expected = evidence[
            "sha256"
        ]

        if observed != expected:
            raise ValueError(
                f"Frozen Phase 4G source-evidence SHA-256 mismatch "
                f"for {key!r}: expected {expected}, "
                f"observed {observed}."
            )


def _expected_model_contract(
    protocol: dict,
    key: str,
) -> dict:
    return dict(
        protocol[
            "final_surrogate"
        ][
            key
        ]
    )


def _validate_final_refit(
    final_refit: Phase4ERFinalRefit,
    protocol: dict,
) -> None:
    """Verify the reconstructed final surrogate is exactly frozen."""

    if final_refit.fit_rows != EXPECTED_FINAL_REFIT_ROWS:
        raise RuntimeError(
            "Phase 4G persistence requires exactly 6,144 final-fit rows."
        )

    if final_refit.test_targets_accessed:
        raise RuntimeError(
            "Phase 4G persistence must not access TEST targets."
        )

    density_expected = _expected_model_contract(
        protocol,
        "density",
    )

    temperature_expected = _expected_model_contract(
        protocol,
        "temperature",
    )

    density = final_refit.density
    temperature = final_refit.temperature

    if (
        density.configuration.target_name
        != density_expected["target"]
    ):
        raise RuntimeError(
            "Unexpected frozen density target."
        )

    if (
        density.configuration.transform_name
        != density_expected["transform"]
    ):
        raise RuntimeError(
            "Unexpected frozen density transform."
        )

    if (
        density.configuration.candidate_spec.candidate_id
        != density_expected["candidate_id"]
    ):
        raise RuntimeError(
            "Unexpected frozen density candidate."
        )

    if not isinstance(
        density.model,
        HistGradientBoostingRegressor,
    ):
        raise RuntimeError(
            "Unexpected frozen density estimator type."
        )

    if (
        temperature.configuration.target_name
        != temperature_expected["target"]
    ):
        raise RuntimeError(
            "Unexpected frozen temperature target."
        )

    if (
        temperature.configuration.transform_name
        != temperature_expected["transform"]
    ):
        raise RuntimeError(
            "Unexpected frozen temperature transform."
        )

    if (
        temperature.configuration.candidate_spec.candidate_id
        != temperature_expected["candidate_id"]
    ):
        raise RuntimeError(
            "Unexpected frozen temperature candidate."
        )

    if not isinstance(
        temperature.model,
        ExtraTreesRegressor,
    ):
        raise RuntimeError(
            "Unexpected frozen temperature estimator type."
        )

    _validate_estimator_parameters(
        density.model,
        density_expected["parameters"],
        model_label="density",
    )

    _validate_estimator_parameters(
        temperature.model,
        temperature_expected["parameters"],
        model_label="temperature",
    )


def _validate_estimator_parameters(
    model: Any,
    expected: dict,
    *,
    model_label: str,
) -> None:
    """Validate all explicitly frozen estimator parameters."""

    observed = model.get_params(
        deep=False
    )

    for name, expected_value in expected.items():
        if name not in observed:
            raise RuntimeError(
                f"{model_label} estimator does not expose "
                f"frozen parameter {name!r}."
            )

        observed_value = observed[
            name
        ]

        if observed_value != expected_value:
            raise RuntimeError(
                f"{model_label} estimator parameter {name!r} "
                f"mismatch: expected {expected_value!r}, "
                f"observed {observed_value!r}."
            )


def _serialize_model(
    model: Any,
    *,
    protocol_number: int,
) -> bytes:
    """Serialize one trusted frozen estimator deterministically."""

    return pickle.dumps(
        model,
        protocol=protocol_number,
    )


def _sha256_bytes(
    payload: bytes,
) -> str:
    import hashlib

    return hashlib.sha256(
        payload
    ).hexdigest()


def _build_manifest(
    *,
    protocol: dict,
    final_refit: Phase4ERFinalRefit,
    density_sha256: str,
    temperature_sha256: str,
    source_git_commit: str,
    repository_clean_state: bool,
    reconstruction_timestamp_utc: str,
    artifact_creation_timestamp_utc: str,
) -> dict:
    """Build the complete persisted-surrogate manifest."""

    runtime = _runtime_metadata()

    density_expected = _expected_model_contract(
        protocol,
        "density",
    )

    temperature_expected = _expected_model_contract(
        protocol,
        "temperature",
    )

    return {
        "phase": "4G",
        "artifact": "persisted_surrogate_manifest",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "serialization": {
            "format": "pickle",
            "pickle_protocol": int(
                protocol[
                    "persistence"
                ][
                    "pickle_protocol"
                ]
            ),
            "trusted_project_artifacts_only": True,
            "hash_verified_before_deserialization": True,
        },
        "feature_contract": {
            "feature_names": list(
                protocol[
                    "final_surrogate"
                ][
                    "feature_order"
                ]
            ),
            "feature_units": dict(
                protocol[
                    "final_surrogate"
                ][
                    "feature_units"
                ]
            ),
            "feature_order_frozen": True,
        },
        "qualified_domain": dict(
            protocol[
                "final_surrogate"
            ][
                "qualified_domain"
            ]
        ),
        "final_fit": {
            "splits": list(
                protocol[
                    "final_surrogate"
                ][
                    "final_fit_splits"
                ]
            ),
            "rows": int(
                final_refit.fit_rows
            ),
            "test_targets_accessed": False,
        },
        "models": {
            "density": {
                "artifact_file": DENSITY_MODEL_FILENAME,
                "sha256": density_sha256,
                "candidate_id": density_expected[
                    "candidate_id"
                ],
                "model_type": density_expected[
                    "model"
                ],
                "target": density_expected[
                    "target"
                ],
                "target_unit": density_expected[
                    "unit"
                ],
                "target_transform": density_expected[
                    "transform"
                ],
                "parameters": dict(
                    density_expected[
                        "parameters"
                    ]
                ),
            },
            "temperature": {
                "artifact_file": TEMPERATURE_MODEL_FILENAME,
                "sha256": temperature_sha256,
                "candidate_id": temperature_expected[
                    "candidate_id"
                ],
                "model_type": temperature_expected[
                    "model"
                ],
                "target": temperature_expected[
                    "target"
                ],
                "target_unit": temperature_expected[
                    "unit"
                ],
                "target_transform": temperature_expected[
                    "transform"
                ],
                "parameters": dict(
                    temperature_expected[
                        "parameters"
                    ]
                ),
            },
        },
        "source_evidence": json.loads(
            json.dumps(
                protocol[
                    "source_evidence"
                ]
            )
        ),
        "provenance": {
            "source_git_commit": source_git_commit,
            "repository_clean_state_before_generation": (
                repository_clean_state
            ),
            **runtime,
            "training_reconstruction_timestamp_utc": (
                reconstruction_timestamp_utc
            ),
            "artifact_creation_timestamp_utc": (
                artifact_creation_timestamp_utc
            ),
        },
        "scientific_limitations": list(
            protocol[
                "scientific_claim_boundaries"
            ]
        ),
    }


def write_phase4g_persisted_surrogate(
    artifact_directory: str | Path = DEFAULT_ARTIFACT_DIRECTORY,
    *,
    protocol_path: str | Path = DEFAULT_PHASE4G_PROTOCOL_PATH,
) -> Path:
    """Reconstruct and persist the frozen surrogate.

    The repository must be clean before production artifact generation.
    Existing production artifact files are never overwritten.
    """

    protocol = load_phase4g_protocol(
        protocol_path
    )

    _validate_source_evidence(
        protocol
    )

    git_commit, repository_clean = _git_state()

    if not repository_clean:
        raise RuntimeError(
            "Phase 4G production artifacts may only be generated "
            "from a clean committed repository state."
        )

    destination = Path(
        artifact_directory
    )

    density_path = (
        destination
        / DENSITY_MODEL_FILENAME
    )

    temperature_path = (
        destination
        / TEMPERATURE_MODEL_FILENAME
    )

    manifest_path = (
        destination
        / DEFAULT_MANIFEST_PATH.name
    )

    for path in (
        density_path,
        temperature_path,
        manifest_path,
    ):
        if path.exists():
            raise FileExistsError(
                "Phase 4G persistence refuses to overwrite "
                f"existing artifact: {path}"
            )

    final_refit = run_phase4er_final_refit()

    _validate_final_refit(
        final_refit,
        protocol,
    )

    reconstruction_timestamp = _utc_now()

    pickle_protocol = int(
        protocol[
            "persistence"
        ][
            "pickle_protocol"
        ]
    )

    density_bytes = _serialize_model(
        final_refit.density.model,
        protocol_number=pickle_protocol,
    )

    temperature_bytes = _serialize_model(
        final_refit.temperature.model,
        protocol_number=pickle_protocol,
    )

    density_sha256 = _sha256_bytes(
        density_bytes
    )

    temperature_sha256 = _sha256_bytes(
        temperature_bytes
    )

    creation_timestamp = _utc_now()

    manifest = _build_manifest(
        protocol=protocol,
        final_refit=final_refit,
        density_sha256=density_sha256,
        temperature_sha256=temperature_sha256,
        source_git_commit=git_commit,
        repository_clean_state=repository_clean,
        reconstruction_timestamp_utc=reconstruction_timestamp,
        artifact_creation_timestamp_utc=creation_timestamp,
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    density_path.write_bytes(
        density_bytes
    )

    temperature_path.write_bytes(
        temperature_bytes
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return manifest_path


def _validate_manifest_against_protocol(
    manifest: dict,
    protocol: dict,
) -> None:
    """Validate frozen metadata before any pickle is deserialized."""

    if manifest.get("phase") != "4G":
        raise ValueError(
            "Persisted surrogate manifest has an unexpected phase."
        )

    if manifest.get("artifact") != "persisted_surrogate_manifest":
        raise ValueError(
            "Unexpected Phase 4G persisted artifact type."
        )

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported Phase 4G manifest schema version."
        )

    serialization = manifest[
        "serialization"
    ]

    if serialization.get(
        "format"
    ) != "pickle":
        raise ValueError(
            "Unexpected persisted surrogate format."
        )

    if int(
        serialization.get(
            "pickle_protocol",
            -1,
        )
    ) != int(
        protocol[
            "persistence"
        ][
            "pickle_protocol"
        ]
    ):
        raise ValueError(
            "Persisted surrogate pickle protocol does not "
            "match the frozen Phase 4G contract."
        )

    feature_contract = manifest[
        "feature_contract"
    ]

    if tuple(
        feature_contract[
            "feature_names"
        ]
    ) != tuple(
        protocol[
            "final_surrogate"
        ][
            "feature_order"
        ]
    ):
        raise ValueError(
            "Persisted surrogate feature order does not "
            "match the frozen Phase 4G contract."
        )

    if feature_contract.get(
        "feature_order_frozen"
    ) is not True:
        raise ValueError(
            "Persisted surrogate manifest does not mark "
            "feature order as frozen."
        )

    if int(
        manifest[
            "final_fit"
        ][
            "rows"
        ]
    ) != EXPECTED_FINAL_REFIT_ROWS:
        raise ValueError(
            "Persisted surrogate final-fit row count is invalid."
        )

    if manifest[
        "final_fit"
    ].get(
        "test_targets_accessed"
    ) is not False:
        raise ValueError(
            "Persisted Phase 4G surrogate must not depend on TEST targets."
        )

    if manifest[
        "source_evidence"
    ] != protocol[
        "source_evidence"
    ]:
        raise ValueError(
            "Persisted surrogate source-evidence metadata "
            "does not match the frozen Phase 4G contract."
        )

    for key in (
        "density",
        "temperature",
    ):
        observed = manifest[
            "models"
        ][
            key
        ]

        expected = _expected_model_contract(
            protocol,
            key,
        )

        checks = {
            "candidate_id": expected[
                "candidate_id"
            ],
            "model_type": expected[
                "model"
            ],
            "target": expected[
                "target"
            ],
            "target_unit": expected[
                "unit"
            ],
            "target_transform": expected[
                "transform"
            ],
            "parameters": expected[
                "parameters"
            ],
        }

        for name, expected_value in checks.items():
            if observed.get(
                name
            ) != expected_value:
                raise ValueError(
                    f"Persisted {key} metadata field {name!r} "
                    "does not match the frozen Phase 4G contract."
                )


def _validate_runtime_compatibility(
    manifest: dict,
) -> None:
    """Require the exact persisted scientific Python runtime versions."""

    current = _runtime_metadata()

    persisted = manifest[
        "provenance"
    ]

    required = (
        "python_version",
        "numpy_version",
        "scipy_version",
        "scikit_learn_version",
    )

    for key in required:
        if persisted.get(
            key
        ) != current[
            key
        ]:
            raise RuntimeError(
                f"Persisted surrogate runtime mismatch for {key}: "
                f"expected {persisted.get(key)!r}, "
                f"observed {current[key]!r}."
            )


def _read_verified_pickle_bytes(
    path: Path,
    *,
    expected_sha256: str,
) -> bytes:
    """Verify artifact bytes before any pickle deserialization occurs."""

    if not path.exists():
        raise FileNotFoundError(
            f"Persisted model artifact not found: {path}"
        )

    payload = path.read_bytes()

    observed = _sha256_bytes(
        payload
    )

    if observed != expected_sha256:
        raise ValueError(
            "Persisted model artifact SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {observed}."
        )

    return payload


def load_phase4g_surrogate(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    protocol_path: str | Path = DEFAULT_PHASE4G_PROTOCOL_PATH,
) -> LoadedPhase4GSurrogate:
    """Load the trusted persisted surrogate after contract/hash validation."""

    protocol = load_phase4g_protocol(
        protocol_path
    )

    manifest_source = Path(
        manifest_path
    )

    manifest = _load_json(
        manifest_source
    )

    _validate_manifest_against_protocol(
        manifest,
        protocol,
    )

    _validate_runtime_compatibility(
        manifest
    )

    density_metadata = manifest[
        "models"
    ][
        "density"
    ]

    temperature_metadata = manifest[
        "models"
    ][
        "temperature"
    ]

    density_path = (
        manifest_source.parent
        / density_metadata[
            "artifact_file"
        ]
    )

    temperature_path = (
        manifest_source.parent
        / temperature_metadata[
            "artifact_file"
        ]
    )

    density_bytes = _read_verified_pickle_bytes(
        density_path,
        expected_sha256=density_metadata[
            "sha256"
        ],
    )

    temperature_bytes = _read_verified_pickle_bytes(
        temperature_path,
        expected_sha256=temperature_metadata[
            "sha256"
        ],
    )

    # Artifacts are trusted project outputs whose exact bytes were verified
    # above. Never use this loader for arbitrary or untrusted pickle files.
    density_model = pickle.loads(
        density_bytes
    )

    temperature_model = pickle.loads(
        temperature_bytes
    )

    if not isinstance(
        density_model,
        HistGradientBoostingRegressor,
    ):
        raise ValueError(
            "Persisted density estimator has an unexpected type."
        )

    if not isinstance(
        temperature_model,
        ExtraTreesRegressor,
    ):
        raise ValueError(
            "Persisted temperature estimator has an unexpected type."
        )

    _validate_estimator_parameters(
        density_model,
        density_metadata[
            "parameters"
        ],
        model_label="density",
    )

    _validate_estimator_parameters(
        temperature_model,
        temperature_metadata[
            "parameters"
        ],
        model_label="temperature",
    )

    return LoadedPhase4GSurrogate(
        density_model=density_model,
        temperature_model=temperature_model,
        density_transform=TargetTransform(
            density_metadata[
                "target_transform"
            ]
        ),
        temperature_transform=TargetTransform(
            temperature_metadata[
                "target_transform"
            ]
        ),
        manifest=manifest,
    )
