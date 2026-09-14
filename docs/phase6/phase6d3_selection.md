# Phase 6D3 ? Optimizer Selection Review and Freeze

## Status

Phase 6D3 reviews the already-frozen Phase 6C and Phase 6D evidence and freezes
the optimizer selection.

Selection artifact:

`configs/phase6/optimizer_selection.json`

SHA-256:

`c4c4963f1f4c0bee2803c4087e514e768a0e95bacd3d58ef1289f887696c3038`

No optimization or surrogate inference was executed during this selection
gate.

## Selection

Primary continuous optimizer:

**Differential Evolution**

Mandatory deterministic reference:

**15,251-point deterministic grid**

Fallback method:

**15,251-point deterministic grid**

SHGO status:

**Not selected / not qualified**

This selection does not yet constitute the final Phase 6 production-runtime
freeze. Phase 6E robustness qualification and Phase 6F runtime freeze remain
required.

## Why Differential Evolution was selected

The frozen Phase 6D2B benchmark contained five qualification scenarios, two
methods and two repeats per method/scenario.

Differential Evolution:

- qualified on all 5/5 scenarios;
- passed both repeats on all five scenarios;
- satisfied exact repeatability on all five scenario pairs;
- returned Phase 6-feasible candidates on every run;
- was the only continuous method to satisfy the complete qualification
  contract.

No tie-break or weighted method-selection score was necessary because only one
continuous method qualified.

## Differential Evolution evidence

| Scenario | Objective | Both runs pass | Exact repeat | Qualifies | Representative continuous-minus-grid objective |
|---|---|---|---|---|---:|
| soft_density_5e16 | density_target_error | True | True | True | 0 |
| hard_density_1e17_min_power | absorbed_power | True | True | True | -0.00621341976979 |
| temperature_window_min_power | absorbed_power | True | True | True | 8.52651282912e-16 |
| boundary_margin_reference | boundary_margin | True | True | True | 1.79356529628e-13 |
| nominal_centre_reference | nominal_distance | True | True | True | 8.98008086147e-14 |

The negative gap for the constrained minimum-power density scenario shows that
the selected continuous method improved the frozen objective relative to the
0.5 W x 0.5 mTorr grid reference.

Beating the grid was not itself required for qualification.

## Why SHGO was not selected

SHGO qualified on 4/5 scenarios but failed:

`hard_density_1e17_min_power`

Both SHGO repeats for that scenario were exactly reproducible, but both failed
the explicit Phase 6 final-feasibility requirement.

Importantly, SciPy reported `success=True` for those failed runs.

This demonstrates why solver status alone is not accepted as evidence of a
scientifically valid operating point.

The frozen Phase 6 feasibility contract takes precedence over the solver
success flag.

SHGO is therefore retained only as a documented global comparator and is not
authorized for post-benchmark retuning within Phase 6D.

## Why the deterministic grid remains

The deterministic grid is not selected as the primary continuous optimizer,
but it remains mandatory as the audit/reference/fallback method.

Its strengths are:

- deterministic behaviour;
- complete enumeration of all 15,251 frozen grid candidates;
- straightforward auditing;
- feasible solutions for all five continuous qualification scenarios;
- no stochastic optimizer behaviour.

Its limitation is resolution: it searches only the frozen 0.5 W by 0.5 mTorr
lattice and therefore does not search the full continuous decision domain.

## Scientific limits

Selection of Differential Evolution does not prove a mathematical global
continuous optimum.

The evidence concerns optimization of the frozen reduced-order plasma
surrogate only.

It does not establish:

- experimental validation;
- industrial validation;
- OIPT hardware performance;
- wafer-scale spatial behaviour;
- reactive chemistry behaviour.

## Governance

The selection was derived only from frozen evidence.

No:

- optimizer parameter was retuned;
- scenario was changed;
- benchmark result was modified;
- deterministic-grid evidence was modified;
- Phase 4 TEST evidence was accessed;
- Phase 5 TEST evidence was accessed;
- source plasma simulator was executed;
- agentic orchestration was used.

## Next step

Phase 6E is unlocked.

Phase 6E must qualify the selected Differential Evolution method under
additional predeclared robustness conditions, including:

- boundary behaviour;
- deliberately infeasible target handling;
- operating-point perturbation sensitivity;
- constraint-edge behaviour;
- comparison against the deterministic grid;
- explicit tradeoff diagnostics.

Phase 6F remains locked until Phase 6E is formally completed.
