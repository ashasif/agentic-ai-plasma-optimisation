"""Phase 4G frozen inference-speed benchmark.

The benchmark specification was frozen before timing at Git commit 771b16c.

This module compares:
- the canonical reduced-order Phase 3 source-simulator workload;
- scalar persisted-surrogate inference;
- batch persisted-surrogate inference.

TEST targets are never accessed here.

Runtime results are hardware and implementation dependent and must not be
interpreted as industrial throughput evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import statistics
import subprocess
import time
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
import scipy
import sklearn

from plasma_ai.experiments.base_dataset import (
    BaseDesignPoint,
    simulate_base_design_point,
)
from plasma_ai.experiments.dataset_config import (
    BaseDatasetConfig,
    load_base_dataset_config,
)
from plasma_ai.surrogate.phase4g_persistence import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PHASE4G_PROTOCOL_PATH,
    LoadedPhase4GSurrogate,
    load_phase4g_protocol,
    load_phase4g_surrogate,
)


DEFAULT_PHASE3_CONFIG_PATH = Path(
    "configs/phase3/base_dataset.json"
)

DEFAULT_OUTPUT_PATH = Path(
    "results/phase4/inference_speed_benchmark.json"
)

BENCHMARK_SPLIT = "phase4g_benchmark"


@dataclass(frozen=True)
class Phase4GBenchmarkGrid:
    """Frozen deterministic 8 x 8 benchmark operating-point grid."""

    features: NDArray[np.float64]

    @property
    def total_points(self) -> int:
        return int(
            self.features.shape[0]
        )


@dataclass(frozen=True)
class WorkloadTiming:
    """Raw and summarized timing results for one workload."""

    repetitions: int
    points_per_workload: int
    raw_runtime_ns: tuple[int, ...]
    median_workload_runtime_ns: float
    minimum_workload_runtime_ns: int
    maximum_workload_runtime_ns: int
    mean_workload_runtime_ns: float
    standard_deviation_ns: float
    median_runtime_per_point_ns: float


def _utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _git_state() -> tuple[str, bool]:
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


def _environment_metadata() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
    }


def build_phase4g_benchmark_grid(
    protocol_path: str | Path = DEFAULT_PHASE4G_PROTOCOL_PATH,
) -> Phase4GBenchmarkGrid:
    """Build the exact frozen 8 x 8 Phase 4G timing grid."""

    protocol = load_phase4g_protocol(
        protocol_path
    )

    grid_spec = protocol[
        "speed_benchmark"
    ][
        "point_grid"
    ]

    if grid_spec.get(
        "type"
    ) != "cartesian_regular":
        raise ValueError(
            "Phase 4G benchmark requires a cartesian_regular grid."
        )

    if grid_spec.get(
        "include_domain_endpoints"
    ) is not True:
        raise ValueError(
            "Phase 4G benchmark requires domain endpoints."
        )

    power_points = int(
        grid_spec[
            "absorbed_power_points"
        ]
    )

    pressure_points = int(
        grid_spec[
            "pressure_points"
        ]
    )

    expected_total = int(
        grid_spec[
            "total_points"
        ]
    )

    if (
        power_points
        * pressure_points
        != expected_total
    ):
        raise ValueError(
            "Phase 4G benchmark grid dimensions do not "
            "match total_points."
        )

    domain = protocol[
        "final_surrogate"
    ][
        "qualified_domain"
    ]

    power_axis = np.linspace(
        float(
            domain[
                "nominal_absorbed_power_W"
            ][
                "min"
            ]
        ),
        float(
            domain[
                "nominal_absorbed_power_W"
            ][
                "max"
            ]
        ),
        power_points,
        endpoint=True,
        dtype=np.float64,
    )

    pressure_axis = np.linspace(
        float(
            domain[
                "target_pressure_mTorr"
            ][
                "min"
            ]
        ),
        float(
            domain[
                "target_pressure_mTorr"
            ][
                "max"
            ]
        ),
        pressure_points,
        endpoint=True,
        dtype=np.float64,
    )

    power_mesh, pressure_mesh = np.meshgrid(
        power_axis,
        pressure_axis,
        indexing="ij",
    )

    features = np.column_stack(
        (
            power_mesh.ravel(),
            pressure_mesh.ravel(),
        )
    ).astype(
        np.float64,
        copy=False,
    )

    if features.shape != (
        expected_total,
        2,
    ):
        raise RuntimeError(
            "Constructed Phase 4G benchmark grid has "
            "an unexpected shape."
        )

    if np.unique(
        features,
        axis=0,
    ).shape[0] != expected_total:
        raise RuntimeError(
            "Constructed Phase 4G benchmark grid "
            "contains duplicate points."
        )

    return Phase4GBenchmarkGrid(
        features=features
    )


def _source_design_point(
    feature_row: NDArray[np.float64],
    *,
    index: int,
    config: BaseDatasetConfig,
) -> BaseDesignPoint:
    """Construct one canonical Phase 3 source operating point."""

    return BaseDesignPoint(
        simulation_id=(
            f"phase4g_benchmark_{index:04d}"
        ),
        split=BENCHMARK_SPLIT,
        split_index=index,
        design_seed=0,
        nominal_absorbed_power_W=float(
            feature_row[0]
        ),
        target_pressure_mTorr=float(
            feature_row[1]
        ),
        nominal_flow_sccm=float(
            config.fixed_flow_sccm
        ),
        gas_temperature_K=float(
            config.gas_temperature_K
        ),
        ion_neutral_cross_section_m2=float(
            config.ion_neutral_cross_section_m2
        ),
        radius_m=float(
            config.radius_m
        ),
        length_m=float(
            config.length_m
        ),
    )


def source_workload(
    features: NDArray[np.float64],
    config: BaseDatasetConfig,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Execute one complete canonical source-simulator workload."""

    density: list[float] = []
    temperature: list[float] = []

    for index, feature_row in enumerate(
        features,
        start=1,
    ):
        point = _source_design_point(
            feature_row,
            index=index,
            config=config,
        )

        row = simulate_base_design_point(
            point,
            config,
        )

        if not row.qualification_valid:
            raise RuntimeError(
                "Phase 4G source benchmark produced a "
                f"qualification-invalid point: {row.simulation_id}"
            )

        if not row.integration_success:
            raise RuntimeError(
                "Phase 4G source benchmark produced an "
                f"integration failure: {row.simulation_id}"
            )

        density.append(
            float(
                row.true_electron_density_m3
            )
        )

        temperature.append(
            float(
                row.true_electron_temperature_eV
            )
        )

    density_array = np.asarray(
        density,
        dtype=np.float64,
    )

    temperature_array = np.asarray(
        temperature,
        dtype=np.float64,
    )

    if not (
        np.isfinite(
            density_array
        ).all()
        and np.isfinite(
            temperature_array
        ).all()
    ):
        raise RuntimeError(
            "Phase 4G source workload produced non-finite targets."
        )

    return (
        density_array,
        temperature_array,
    )


def surrogate_scalar_workload(
    features: NDArray[np.float64],
    surrogate: LoadedPhase4GSurrogate,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Execute one complete scalar persisted-surrogate workload."""

    density: list[float] = []
    temperature: list[float] = []

    for feature_row in features:
        prediction = surrogate.predict_physical(
            np.asarray(
                feature_row,
                dtype=np.float64,
            ).reshape(
                1,
                2,
            )
        )

        density.append(
            float(
                prediction.electron_density_m3[
                    0
                ]
            )
        )

        temperature.append(
            float(
                prediction.electron_temperature_eV[
                    0
                ]
            )
        )

    return (
        np.asarray(
            density,
            dtype=np.float64,
        ),
        np.asarray(
            temperature,
            dtype=np.float64,
        ),
    )


def surrogate_batch_workload(
    features: NDArray[np.float64],
    surrogate: LoadedPhase4GSurrogate,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Execute one complete batch persisted-surrogate workload."""

    prediction = surrogate.predict_physical(
        features
    )

    return (
        np.asarray(
            prediction.electron_density_m3,
            dtype=np.float64,
        ),
        np.asarray(
            prediction.electron_temperature_eV,
            dtype=np.float64,
        ),
    )


def time_workload(
    workload: Callable[[], object],
    *,
    warmup_workloads: int,
    timed_repetitions: int,
    points_per_workload: int,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> WorkloadTiming:
    """Warm up and time a predeclared complete workload."""

    if warmup_workloads < 0:
        raise ValueError(
            "warmup_workloads must be non-negative."
        )

    if timed_repetitions <= 0:
        raise ValueError(
            "timed_repetitions must be positive."
        )

    if points_per_workload <= 0:
        raise ValueError(
            "points_per_workload must be positive."
        )

    for _ in range(
        warmup_workloads
    ):
        workload()

    runtimes: list[int] = []

    for _ in range(
        timed_repetitions
    ):
        start = int(
            clock()
        )

        workload()

        end = int(
            clock()
        )

        runtime = (
            end - start
        )

        if runtime < 0:
            raise RuntimeError(
                "Timing clock produced a negative duration."
            )

        runtimes.append(
            runtime
        )

    median_runtime = float(
        statistics.median(
            runtimes
        )
    )

    mean_runtime = float(
        statistics.fmean(
            runtimes
        )
    )

    standard_deviation = (
        float(
            statistics.pstdev(
                runtimes
            )
        )
        if len(
            runtimes
        ) > 1
        else 0.0
    )

    return WorkloadTiming(
        repetitions=timed_repetitions,
        points_per_workload=points_per_workload,
        raw_runtime_ns=tuple(
            runtimes
        ),
        median_workload_runtime_ns=median_runtime,
        minimum_workload_runtime_ns=int(
            min(
                runtimes
            )
        ),
        maximum_workload_runtime_ns=int(
            max(
                runtimes
            )
        ),
        mean_workload_runtime_ns=mean_runtime,
        standard_deviation_ns=standard_deviation,
        median_runtime_per_point_ns=(
            median_runtime
            / points_per_workload
        ),
    )


def _timing_payload(
    timing: WorkloadTiming,
) -> dict:
    return {
        "repetitions": int(
            timing.repetitions
        ),
        "points_per_workload": int(
            timing.points_per_workload
        ),
        "raw_runtime_ns": [
            int(value)
            for value in timing.raw_runtime_ns
        ],
        "median_workload_runtime_ns": float(
            timing.median_workload_runtime_ns
        ),
        "minimum_workload_runtime_ns": int(
            timing.minimum_workload_runtime_ns
        ),
        "maximum_workload_runtime_ns": int(
            timing.maximum_workload_runtime_ns
        ),
        "mean_workload_runtime_ns": float(
            timing.mean_workload_runtime_ns
        ),
        "standard_deviation_ns": float(
            timing.standard_deviation_ns
        ),
        "median_runtime_per_point_ns": float(
            timing.median_runtime_per_point_ns
        ),
    }


def run_phase4g_speed_benchmark(
    *,
    protocol_path: str | Path = DEFAULT_PHASE4G_PROTOCOL_PATH,
    phase3_config_path: str | Path = DEFAULT_PHASE3_CONFIG_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> dict:
    """Execute the frozen Phase 4G benchmark exactly as predeclared."""

    protocol = load_phase4g_protocol(
        protocol_path
    )

    git_commit, repository_clean = _git_state()

    if not repository_clean:
        raise RuntimeError(
            "Phase 4G production benchmark may only run "
            "from a clean committed repository state."
        )

    grid = build_phase4g_benchmark_grid(
        protocol_path
    )

    config = load_base_dataset_config(
        phase3_config_path
    )

    surrogate = load_phase4g_surrogate(
        manifest_path,
        protocol_path=protocol_path,
    )

    benchmark = protocol[
        "speed_benchmark"
    ]

    expected_points = int(
        benchmark[
            "point_grid"
        ][
            "total_points"
        ]
    )

    if grid.total_points != expected_points:
        raise RuntimeError(
            "Phase 4G benchmark point count does not "
            "match the frozen protocol."
        )

    source_spec = benchmark[
        "source"
    ]

    scalar_spec = benchmark[
        "surrogate_scalar"
    ]

    batch_spec = benchmark[
        "surrogate_batch"
    ]

    source_callable = lambda: source_workload(
        grid.features,
        config,
    )

    scalar_callable = lambda: surrogate_scalar_workload(
        grid.features,
        surrogate,
    )

    batch_callable = lambda: surrogate_batch_workload(
        grid.features,
        surrogate,
    )

    source_timing = time_workload(
        source_callable,
        warmup_workloads=int(
            source_spec[
                "warmup_workloads"
            ]
        ),
        timed_repetitions=int(
            source_spec[
                "timed_repetitions"
            ]
        ),
        points_per_workload=expected_points,
    )

    scalar_timing = time_workload(
        scalar_callable,
        warmup_workloads=int(
            scalar_spec[
                "warmup_workloads"
            ]
        ),
        timed_repetitions=int(
            scalar_spec[
                "timed_repetitions"
            ]
        ),
        points_per_workload=expected_points,
    )

    batch_timing = time_workload(
        batch_callable,
        warmup_workloads=int(
            batch_spec[
                "warmup_workloads"
            ]
        ),
        timed_repetitions=int(
            batch_spec[
                "timed_repetitions"
            ]
        ),
        points_per_workload=expected_points,
    )

    scalar_speedup = (
        source_timing.median_workload_runtime_ns
        / scalar_timing.median_workload_runtime_ns
    )

    batch_speedup = (
        source_timing.median_workload_runtime_ns
        / batch_timing.median_workload_runtime_ns
    )

    return {
        "phase": "4G",
        "stage": "frozen_inference_speed_benchmark",
        "benchmark_status": "completed",
        "test_targets_accessed": False,
        "protocol_status": protocol[
            "status"
        ],
        "git": {
            "commit": git_commit,
            "repository_clean_state_before_benchmark": (
                repository_clean
            ),
        },
        "environment": {
            **_environment_metadata(),
            "benchmark_timestamp_utc": _utc_now(),
        },
        "benchmark_grid": {
            "type": "cartesian_regular",
            "absorbed_power_points": int(
                benchmark[
                    "point_grid"
                ][
                    "absorbed_power_points"
                ]
            ),
            "pressure_points": int(
                benchmark[
                    "point_grid"
                ][
                    "pressure_points"
                ]
            ),
            "total_points": int(
                grid.total_points
            ),
            "include_domain_endpoints": True,
            "feature_order": list(
                protocol[
                    "final_surrogate"
                ][
                    "feature_order"
                ]
            ),
        },
        "timing_protocol": {
            "timing_function": benchmark[
                "timing_function"
            ],
            "primary_summary": benchmark[
                "primary_summary"
            ],
            "source_warmup_workloads": int(
                source_spec[
                    "warmup_workloads"
                ]
            ),
            "source_timed_repetitions": int(
                source_spec[
                    "timed_repetitions"
                ]
            ),
            "surrogate_scalar_warmup_workloads": int(
                scalar_spec[
                    "warmup_workloads"
                ]
            ),
            "surrogate_scalar_timed_repetitions": int(
                scalar_spec[
                    "timed_repetitions"
                ]
            ),
            "surrogate_batch_warmup_workloads": int(
                batch_spec[
                    "warmup_workloads"
                ]
            ),
            "surrogate_batch_timed_repetitions": int(
                batch_spec[
                    "timed_repetitions"
                ]
            ),
        },
        "included_operations": {
            "source": [
                "canonical source operating-point construction",
                "canonical simulate_base_design_point execution",
                "source target extraction",
                "canonical source qualification checks",
            ],
            "surrogate_scalar": [
                "production inference input validation",
                "density prediction",
                "density inverse log10 transformation",
                "temperature prediction",
                "physical-scale result construction",
            ],
            "surrogate_batch": [
                "production batch-input validation",
                "density batch prediction",
                "density inverse log10 transformation",
                "temperature batch prediction",
                "physical-scale result construction",
            ],
        },
        "excluded_operations": [
            "configuration loading",
            "persisted model loading",
            "benchmark-grid construction",
            "JSON serialization",
            "file writing",
            "console output",
        ],
        "timings": {
            "source": _timing_payload(
                source_timing
            ),
            "surrogate_scalar": _timing_payload(
                scalar_timing
            ),
            "surrogate_batch": _timing_payload(
                batch_timing
            ),
        },
        "speedup": {
            "scalar": float(
                scalar_speedup
            ),
            "batch": float(
                batch_speedup
            ),
            "definition": (
                "median source 64-point workload runtime divided "
                "by median corresponding surrogate 64-point "
                "workload runtime"
            ),
        },
        "interpretation": {
            "hardware_and_implementation_dependent": True,
            "industrial_throughput_claim": False,
            "model_selection_evidence": False,
            "test_evaluation": False,
            "protocol_changed_after_timing": False,
        },
        "scientific_scope": {
            "synthetic_data": True,
            "reduced_order_argon_plasma_model": True,
            "numerically_qualified_model_envelope": True,
            "surrogate_of_reduced_order_simulator": True,
            "experimental_validation": False,
            "industrial_validation": False,
            "oipt_operating_range_claim": False,
            "absorbed_power_is_generator_rf_power": False,
            "reactive_etch_or_deposition_prediction": False,
            "wafer_scale_spatial_modelling": False,
        },
    }


def write_phase4g_speed_benchmark(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    protocol_path: str | Path = DEFAULT_PHASE4G_PROTOCOL_PATH,
    phase3_config_path: str | Path = DEFAULT_PHASE3_CONFIG_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> Path:
    """Execute and write the frozen benchmark without overwrite."""

    destination = Path(
        output_path
    )

    if destination.exists():
        raise FileExistsError(
            "Phase 4G benchmark artifact already exists. "
            "Refusing to overwrite frozen timing evidence."
        )

    payload = run_phase4g_speed_benchmark(
        protocol_path=protocol_path,
        phase3_config_path=phase3_config_path,
        manifest_path=manifest_path,
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return destination
