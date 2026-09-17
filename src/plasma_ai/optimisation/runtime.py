"""Qualified Phase 6 production optimisation runtime."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from plasma_ai.optimisation.grid import (
    DeterministicGrid,
    GridScenarioEvaluation,
    build_deterministic_grid,
    evaluate_grid_scenario,
    load_frozen_grid_scenario_set,
)
from plasma_ai.optimisation.primitives import (
    CandidateEvaluation,
)
from plasma_ai.optimisation.protocol import (
    EffectivePhase6Protocol,
    load_effective_phase6_protocol,
)
from plasma_ai.optimisation.robustness import (
    FrozenRobustnessProtocol,
    SeededDERunResult,
    load_robustness_qualification_protocol,
    run_seeded_differential_evolution,
)
from plasma_ai.optimisation.runtime_manifest import (
    EffectiveRuntimeProtocol,
    RuntimeManifest,
    RuntimeManifestError,
    load_effective_runtime_protocol,
    load_runtime_manifest,
)
from plasma_ai.optimisation.scenario import (
    OperatingPoint,
    OptimisationScenario,
    parse_optimisation_scenario,
)
from plasma_ai.optimisation.surrogate_adapter import (
    Phase6SurrogateAdapter,
    load_phase6_surrogate_adapter,
)


class ProductionRuntimeError(RuntimeError):
    """Raised when the frozen Phase 6 runtime contract is violated."""


@dataclass(frozen=True)
class RuntimeSelectedMethodResult:
    """Serializable selected-method runtime evidence."""

    run_pass: bool
    final_phase6_feasible: bool
    decision_coordinates: tuple[float, float] | None
    objective_name: str
    objective_value: float | None
    electron_density_m3: float | None
    electron_temperature_eV: float | None
    scipy_success: bool
    scipy_message: str
    constraint_values: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class RuntimeGridReference:
    """Serializable deterministic-grid runtime reference."""

    feasible_candidate_count: int
    selected_candidate_index: int | None
    operating_point: OperatingPoint | None
    electron_density_m3: float | None
    electron_temperature_eV: float | None
    primary_objective: float | None
    scenario_feasible: bool


@dataclass(frozen=True)
class RuntimeResponse:
    """Frozen non-timing Phase 6 runtime response."""

    status: str
    scenario_id: str
    chosen_source: str | None
    chosen_operating_point: OperatingPoint | None
    chosen_electron_density_m3: float | None
    chosen_electron_temperature_eV: float | None
    chosen_primary_objective: float | None
    selected_method_result: RuntimeSelectedMethodResult
    grid_reference_result: RuntimeGridReference
    selected_minus_grid_primary_objective: float | None
    provenance: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return canonical non-timing serializable response."""

        def point_payload(
            point: OperatingPoint | None,
        ):
            if point is None:
                return None

            return {
                "nominal_absorbed_power_W": float(
                    point.nominal_absorbed_power_W
                ),
                "target_pressure_mTorr": float(
                    point.target_pressure_mTorr
                ),
            }

        selected = self.selected_method_result
        grid = self.grid_reference_result

        return {
            "status": self.status,
            "scenario_id": self.scenario_id,
            "chosen_source": self.chosen_source,
            "chosen_operating_point": point_payload(
                self.chosen_operating_point
            ),
            "chosen_predictions": (
                None
                if self.chosen_electron_density_m3 is None
                else {
                    "electron_density_m3": float(
                        self.chosen_electron_density_m3
                    ),
                    "electron_temperature_eV": float(
                        self.chosen_electron_temperature_eV
                    ),
                }
            ),
            "chosen_primary_objective": (
                self.chosen_primary_objective
            ),
            "selected_method_result": {
                "run_pass": selected.run_pass,
                "final_phase6_feasible": (
                    selected.final_phase6_feasible
                ),
                "decision_coordinates": (
                    None
                    if selected.decision_coordinates is None
                    else {
                        "nominal_absorbed_power_W": float(
                            selected.decision_coordinates[0]
                        ),
                        "target_pressure_mTorr": float(
                            selected.decision_coordinates[1]
                        ),
                    }
                ),
                "objective_name": (
                    selected.objective_name
                ),
                "objective_value": (
                    selected.objective_value
                ),
                "electron_density_m3": (
                    selected.electron_density_m3
                ),
                "electron_temperature_eV": (
                    selected.electron_temperature_eV
                ),
                "scipy_success": (
                    selected.scipy_success
                ),
                "scipy_message": (
                    selected.scipy_message
                ),
                "constraint_values": [
                    {
                        "name": name,
                        "value": float(value),
                    }
                    for name, value
                    in selected.constraint_values
                ],
            },
            "grid_reference_result": {
                "feasible_candidate_count": (
                    grid.feasible_candidate_count
                ),
                "selected_candidate_index": (
                    grid.selected_candidate_index
                ),
                "operating_point": point_payload(
                    grid.operating_point
                ),
                "electron_density_m3": (
                    grid.electron_density_m3
                ),
                "electron_temperature_eV": (
                    grid.electron_temperature_eV
                ),
                "primary_objective": (
                    grid.primary_objective
                ),
                "scenario_feasible": (
                    grid.scenario_feasible
                ),
            },
            "selected_minus_grid_primary_objective": (
                self.selected_minus_grid_primary_objective
            ),
            "provenance": {
                key: value
                for key, value
                in self.provenance
            },
        }


SelectedRunner = Callable[
    [OptimisationScenario],
    SeededDERunResult,
]


def array_sha256(
    values,
) -> str:
    """Hash a float64 C-contiguous array exactly."""

    array = np.ascontiguousarray(
        np.asarray(
            values,
            dtype=np.float64,
        )
    )

    return hashlib.sha256(
        array.tobytes(
            order="C"
        )
    ).hexdigest()


def _primary_objective(
    evaluation: CandidateEvaluation,
    scenario: OptimisationScenario,
) -> float:
    objective = scenario.objective_priority[0]

    mapping = {
        "density_target_error":
            "density_relative_error",
        "absorbed_power":
            "normalized_absorbed_power",
        "boundary_margin":
            "boundary_margin_cost",
        "nominal_distance":
            "normalized_nominal_distance",
    }

    try:
        metric = mapping[
            objective
        ]
    except KeyError as exc:
        raise ProductionRuntimeError(
            f"Unsupported runtime objective {objective!r}."
        ) from exc

    try:
        value = float(
            evaluation.metrics[
                metric
            ]
        )
    except KeyError as exc:
        raise ProductionRuntimeError(
            f"Missing runtime objective metric {metric!r}."
        ) from exc

    if not math.isfinite(
        value
    ):
        raise ProductionRuntimeError(
            "Runtime primary objective is non-finite."
        )

    return value


def _selected_payload(
    result: SeededDERunResult,
) -> RuntimeSelectedMethodResult:
    return RuntimeSelectedMethodResult(
        run_pass=bool(
            result.run_pass
        ),
        final_phase6_feasible=bool(
            result.final_phase6_feasible
        ),
        decision_coordinates=(
            None
            if result.decision_coordinates is None
            else (
                float(
                    result.decision_coordinates[0]
                ),
                float(
                    result.decision_coordinates[1]
                ),
            )
        ),
        objective_name=result.objective_name,
        objective_value=(
            None
            if result.objective_value is None
            else float(
                result.objective_value
            )
        ),
        electron_density_m3=(
            None
            if result.electron_density_m3 is None
            else float(
                result.electron_density_m3
            )
        ),
        electron_temperature_eV=(
            None
            if result.electron_temperature_eV is None
            else float(
                result.electron_temperature_eV
            )
        ),
        scipy_success=bool(
            result.scipy_success
        ),
        scipy_message=str(
            result.scipy_message
        ),
        constraint_values=tuple(
            (
                str(name),
                float(value),
            )
            for name, value
            in result.constraint_values
        ),
    )


def _grid_payload(
    result: GridScenarioEvaluation,
) -> RuntimeGridReference:

    evaluation = (
        result.selected_evaluation
    )

    if (
        result.selected_candidate_index
        is None
    ):
        if evaluation is not None:
            raise ProductionRuntimeError(
                "Grid evaluation contains inconsistent empty selection."
            )

        return RuntimeGridReference(
            feasible_candidate_count=int(
                result.feasible_candidate_count
            ),
            selected_candidate_index=None,
            operating_point=None,
            electron_density_m3=None,
            electron_temperature_eV=None,
            primary_objective=None,
            scenario_feasible=False,
        )

    if evaluation is None:
        raise ProductionRuntimeError(
            "Grid selected index exists without evaluation."
        )

    if result.selected_objective_key is None:
        raise ProductionRuntimeError(
            "Grid selected candidate lacks objective key."
        )

    if len(
        result.selected_objective_key
    ) < 1:
        raise ProductionRuntimeError(
            "Grid selected objective key is empty."
        )

    return RuntimeGridReference(
        feasible_candidate_count=int(
            result.feasible_candidate_count
        ),
        selected_candidate_index=int(
            result.selected_candidate_index
        ),
        operating_point=(
            evaluation.operating_point
        ),
        electron_density_m3=float(
            evaluation.electron_density_m3
        ),
        electron_temperature_eV=float(
            evaluation.electron_temperature_eV
        ),
        primary_objective=float(
            result.selected_objective_key[0]
        ),
        scenario_feasible=bool(
            evaluation.scenario_feasible
        ),
    )


@dataclass(frozen=True)
class Phase6OptimisationRuntime:
    """Initialized Phase 6 production runtime core."""

    protocol: EffectivePhase6Protocol
    runtime_protocol: EffectiveRuntimeProtocol
    robustness_protocol: FrozenRobustnessProtocol

    grid: DeterministicGrid
    grid_density_m3: NDArray[np.float64]
    grid_temperature_eV: NDArray[np.float64]

    selected_runner: SelectedRunner

    provenance: tuple[tuple[str, str], ...]

    def run(
        self,
        scenario: OptimisationScenario,
    ) -> RuntimeResponse:
        """Execute one frozen single-objective runtime request."""

        if len(
            scenario.objective_priority
        ) != 1:
            raise ProductionRuntimeError(
                "Production runtime does not support multiobjective requests."
            )

        grid_result = evaluate_grid_scenario(
            self.grid,
            electron_density_m3=self.grid_density_m3,
            electron_temperature_eV=self.grid_temperature_eV,
            scenario=scenario,
            protocol=self.protocol,
        )

        grid_payload = _grid_payload(
            grid_result
        )

        selected_raw = self.selected_runner(
            scenario
        )

        selected = _selected_payload(
            selected_raw
        )

        selected_feasible = bool(
            selected.run_pass
            and selected.final_phase6_feasible
            and selected.objective_value is not None
            and math.isfinite(
                selected.objective_value
            )
        )

        grid_feasible = bool(
            grid_payload.feasible_candidate_count > 0
            and grid_payload.selected_candidate_index is not None
            and grid_payload.scenario_feasible
            and grid_payload.primary_objective is not None
            and math.isfinite(
                grid_payload.primary_objective
            )
        )

        tolerance = float(
            self.runtime_protocol.base[
                "grid_reference_contract"
            ][
                "grid_reference_gap_tolerance"
            ]
        )

        status: str
        chosen_source: str | None
        chosen_point: OperatingPoint | None
        chosen_density: float | None
        chosen_temperature: float | None
        chosen_objective: float | None
        gap: float | None

        if selected_feasible and not grid_feasible:

            status = (
                "selected_method_accepted_grid_infeasible"
            )

            chosen_source = (
                "differential_evolution"
            )

            assert (
                selected.decision_coordinates
                is not None
            )

            chosen_point = OperatingPoint(
                nominal_absorbed_power_W=float(
                    selected.decision_coordinates[0]
                ),
                target_pressure_mTorr=float(
                    selected.decision_coordinates[1]
                ),
            )

            chosen_density = (
                selected.electron_density_m3
            )

            chosen_temperature = (
                selected.electron_temperature_eV
            )

            chosen_objective = (
                selected.objective_value
            )

            gap = None

        elif selected_feasible and grid_feasible:

            assert (
                selected.objective_value
                is not None
            )

            assert (
                grid_payload.primary_objective
                is not None
            )

            gap = float(
                selected.objective_value
                - grid_payload.primary_objective
            )

            if (
                selected.objective_value
                <= (
                    grid_payload.primary_objective
                    + tolerance
                )
            ):

                status = (
                    "selected_method_accepted"
                )

                chosen_source = (
                    "differential_evolution"
                )

                assert (
                    selected.decision_coordinates
                    is not None
                )

                chosen_point = OperatingPoint(
                    nominal_absorbed_power_W=float(
                        selected.decision_coordinates[0]
                    ),
                    target_pressure_mTorr=float(
                        selected.decision_coordinates[1]
                    ),
                )

                chosen_density = (
                    selected.electron_density_m3
                )

                chosen_temperature = (
                    selected.electron_temperature_eV
                )

                chosen_objective = (
                    selected.objective_value
                )

            else:

                status = (
                    "grid_fallback_selected_method_objective_regression"
                )

                chosen_source = (
                    "deterministic_grid"
                )

                chosen_point = (
                    grid_payload.operating_point
                )

                chosen_density = (
                    grid_payload.electron_density_m3
                )

                chosen_temperature = (
                    grid_payload.electron_temperature_eV
                )

                chosen_objective = (
                    grid_payload.primary_objective
                )

        elif (
            (not selected_feasible)
            and grid_feasible
        ):

            status = (
                "grid_fallback_selected_method_infeasible"
            )

            chosen_source = (
                "deterministic_grid"
            )

            chosen_point = (
                grid_payload.operating_point
            )

            chosen_density = (
                grid_payload.electron_density_m3
            )

            chosen_temperature = (
                grid_payload.electron_temperature_eV
            )

            chosen_objective = (
                grid_payload.primary_objective
            )

            gap = None

        else:

            status = (
                "no_feasible_point_found_under_search_protocol"
            )

            chosen_source = None
            chosen_point = None
            chosen_density = None
            chosen_temperature = None
            chosen_objective = None
            gap = None

        allowed_statuses = tuple(
            self.runtime_protocol.amendment[
                "effective_status_vocabulary"
            ]
        )

        if status not in allowed_statuses:
            raise ProductionRuntimeError(
                "Runtime produced an unqualified status."
            )

        return RuntimeResponse(
            status=status,
            scenario_id=scenario.scenario_id,
            chosen_source=chosen_source,
            chosen_operating_point=chosen_point,
            chosen_electron_density_m3=chosen_density,
            chosen_electron_temperature_eV=chosen_temperature,
            chosen_primary_objective=chosen_objective,
            selected_method_result=selected,
            grid_reference_result=grid_payload,
            selected_minus_grid_primary_objective=gap,
            provenance=self.provenance,
        )


def build_runtime_from_components(
    *,
    grid: DeterministicGrid,
    grid_density_m3,
    grid_temperature_eV,
    selected_runner: SelectedRunner,
    provenance: tuple[tuple[str, str], ...] = (),
    protocol: EffectivePhase6Protocol | None = None,
    runtime_protocol: EffectiveRuntimeProtocol | None = None,
    robustness_protocol: FrozenRobustnessProtocol | None = None,
) -> Phase6OptimisationRuntime:
    """Build runtime core from already-qualified components.

    Intended for synthetic qualification and internal production loader use.
    """

    if protocol is None:
        protocol = (
            load_effective_phase6_protocol()
        )

    if runtime_protocol is None:
        runtime_protocol = (
            load_effective_runtime_protocol()
        )

    if robustness_protocol is None:
        robustness_protocol = (
            load_robustness_qualification_protocol()
        )

    density = np.asarray(
        grid_density_m3,
        dtype=np.float64,
    ).copy()

    temperature = np.asarray(
        grid_temperature_eV,
        dtype=np.float64,
    ).copy()

    rows = grid.input_matrix.shape[0]

    if density.shape != (
        rows,
    ):
        raise ProductionRuntimeError(
            "Runtime grid density vector shape mismatch."
        )

    if temperature.shape != (
        rows,
    ):
        raise ProductionRuntimeError(
            "Runtime grid temperature vector shape mismatch."
        )

    if not np.all(
        np.isfinite(
            density
        )
    ):
        raise ProductionRuntimeError(
            "Runtime grid density predictions are non-finite."
        )

    if not np.all(
        np.isfinite(
            temperature
        )
    ):
        raise ProductionRuntimeError(
            "Runtime grid temperature predictions are non-finite."
        )

    if not np.all(
        density > 0.0
    ):
        raise ProductionRuntimeError(
            "Runtime grid density predictions must be positive."
        )

    if not np.all(
        temperature > 0.0
    ):
        raise ProductionRuntimeError(
            "Runtime grid temperature predictions must be positive."
        )

    density.setflags(
        write=False
    )

    temperature.setflags(
        write=False
    )

    return Phase6OptimisationRuntime(
        protocol=protocol,
        runtime_protocol=runtime_protocol,
        robustness_protocol=robustness_protocol,
        grid=grid,
        grid_density_m3=density,
        grid_temperature_eV=temperature,
        selected_runner=selected_runner,
        provenance=tuple(
            provenance
        ),
    )


def load_phase6_optimisation_runtime(
    manifest_path: str = (
        "artifacts/phase6/"
        "optimization_runtime_manifest.json"
    ),
) -> Phase6OptimisationRuntime:
    """Load the real frozen Phase 6 production runtime."""

    manifest: RuntimeManifest = (
        load_runtime_manifest(
            manifest_path
        )
    )

    protocol = (
        load_effective_phase6_protocol()
    )

    runtime_protocol = (
        load_effective_runtime_protocol()
    )

    robustness_protocol = (
        load_robustness_qualification_protocol()
    )

    scenario_set = (
        load_frozen_grid_scenario_set(
            protocol=protocol,
        )
    )

    grid = build_deterministic_grid(
        scenario_set,
        protocol=protocol,
    )

    adapter: Phase6SurrogateAdapter = (
        load_phase6_surrogate_adapter()
    )

    batch = adapter.predict_batch(
        grid.input_matrix
    )

    expected_input_hash = manifest.payload[
        "grid_input_array_sha256"
    ]

    expected_density_hash = manifest.payload[
        "grid_density_array_sha256"
    ]

    expected_temperature_hash = manifest.payload[
        "grid_temperature_array_sha256"
    ]

    observed_input_hash = array_sha256(
        batch.input_matrix
    )

    observed_density_hash = array_sha256(
        batch.electron_density_m3
    )

    observed_temperature_hash = array_sha256(
        batch.electron_temperature_eV
    )

    if observed_input_hash != expected_input_hash:
        raise RuntimeManifestError(
            "Runtime grid input-array hash mismatch."
        )

    if observed_density_hash != expected_density_hash:
        raise RuntimeManifestError(
            "Runtime grid density-array hash mismatch."
        )

    if observed_temperature_hash != expected_temperature_hash:
        raise RuntimeManifestError(
            "Runtime grid temperature-array hash mismatch."
        )

    production_seed = int(
        runtime_protocol.base[
            "runtime_identity"
        ][
            "production_seed"
        ]
    )

    def selected_runner(
        scenario: OptimisationScenario,
    ) -> SeededDERunResult:

        return run_seeded_differential_evolution(
            scenario,
            seed=production_seed,
            predictor=adapter.predict_point,
            robustness=robustness_protocol,
            protocol=protocol,
        )

    provenance = (
        (
            "runtime_manifest_sha256",
            manifest.sha256,
        ),
        (
            "effective_runtime_protocol_sha256",
            runtime_protocol.effective_sha256,
        ),
        (
            "phase4g_surrogate_manifest_sha256",
            adapter.manifest_sha256,
        ),
    )

    return build_runtime_from_components(
        grid=grid,
        grid_density_m3=(
            batch.electron_density_m3
        ),
        grid_temperature_eV=(
            batch.electron_temperature_eV
        ),
        selected_runner=selected_runner,
        provenance=provenance,
        protocol=protocol,
        runtime_protocol=runtime_protocol,
        robustness_protocol=robustness_protocol,
    )


def parse_runtime_request(
    raw: dict[str, Any],
    *,
    protocol: EffectivePhase6Protocol | None = None,
) -> OptimisationScenario:
    """Parse and enforce production runtime request scope."""

    if protocol is None:
        protocol = (
            load_effective_phase6_protocol()
        )

    scenario = parse_optimisation_scenario(
        raw,
        protocol=protocol,
    )

    if len(
        scenario.objective_priority
    ) != 1:
        raise ProductionRuntimeError(
            "Production runtime does not support multiobjective requests."
        )

    return scenario
