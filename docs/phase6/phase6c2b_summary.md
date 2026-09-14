# Phase 6C2B ? Deterministic Grid Baseline Results

## Status

Phase 6C2B is the first executed Phase 6 operating-point optimisation
baseline.

The frozen deterministic grid was evaluated using the committed Phase 6C2A
engine and the trusted frozen Phase 4G surrogate.

Result:

`results/phase6/deterministic_grid_baseline.json`

SHA-256:

`cb3cbad2345d5a83286e7940fdd7f1960e96b1a3fdd7eea6f0492caa058a7f59`

Execution source commit:

`7b065c753c8893d29714b1d1e6e6a2eee0fd2f0c`

## Execution contract

The baseline contains exactly 15,251 candidate operating points:

- absorbed power: 15?90 W inclusive;
- absorbed-power increment: 0.5 W;
- pressure: 10?60 mTorr inclusive;
- pressure increment: 0.5 mTorr.

Exactly one complete-grid surrogate batch call was made.

The resulting density and temperature arrays were reused unchanged across all
ten frozen qualification scenarios.

Scenario evaluation was then repeated against the same prediction arrays and
returned exact equality.

## Prediction envelope on the frozen grid

Electron-density surrogate predictions:

- minimum: `16197742107743246` m^-3
- maximum: `2.2900906167162906e+17` m^-3
- mean: `1.0204769575982042e+17` m^-3

Electron-temperature surrogate predictions:

- minimum: `1.5473200462621457` eV
- maximum: `2.0744792075315606` eV
- mean: `1.7145654894635056` eV

These are surrogate predictions across the qualified grid. They are not new
source-model, experimental or industrial measurements.

## Frozen scenario outcomes

| Scenario | Feasible grid points | Selected index | Power W | Pressure mTorr | Predicted density m^-3 | Predicted Te eV | Pareto count |
|---|---:|---:|---:|---:|---:|---:|---:|
| soft_density_5e16 | 15251 | 1680 | 23 | 42 | 5.000722e+16 | 1.63075 | 0 |
| soft_density_1e17_low_power | 15251 | 6525 | 47 | 40.5 | 9.999810e+16 | 1.63976 | 53 |
| soft_density_2e17_low_power | 15251 | 13624 | 82 | 55 | 1.999996e+17 | 1.56687 | 118 |
| hard_density_1e17_min_power | 1136 | 4940 | 39 | 56 | 9.514427e+16 | 1.56277 | 0 |
| hard_density_1p5e17_nominal | 1061 | 10363 | 66 | 40.5 | 1.433760e+17 | 1.63976 | 11 |
| temperature_window_min_power | 5889 | 19 | 15 | 19.5 | 2.358463e+16 | 1.84306 | 0 |
| density_1p5e17_temperature_window | 5889 | 13469 | 81.5 | 28 | 1.500095e+17 | 1.73677 | 120 |
| hard_density_interior_margin | 1763 | 7625 | 52.5 | 35 | 1.034105e+17 | 1.67686 | 193 |
| boundary_margin_reference | 15251 | 7625 | 52.5 | 35 | 1.034105e+17 | 1.67686 | 0 |
| nominal_centre_reference | 15251 | 7625 | 52.5 | 35 | 1.034105e+17 | 1.67686 | 0 |

## Interpretation boundary

Every selected operating point is a **discrete grid-baseline result**.

The grid spacing is 0.5 W by 0.5 mTorr. Therefore these results do not establish
the continuous optimum.

A scenario with no feasible grid candidate would be reported only as:

`no_feasible_point_found_under_search_protocol`

That status would not prove global physical infeasibility.

## Multiobjective evidence

Where more than one objective is active:

- hard constraints are applied before ranking;
- declared objectives are ranked lexicographically;
- no hidden weighted scalarisation is used;
- exact non-dominated grid candidates are retained;
- no numerical Pareto tolerance is introduced.

## Evidence separation

This result used:

- the frozen production Phase 4G surrogate;
- the frozen Phase 6 effective contract;
- the frozen Phase 6C1 qualification scenarios;
- the committed Phase 6C2A grid engine.

It did not use:

- Phase 4 TEST targets;
- the Phase 5 TEST dataset;
- the source plasma simulator;
- experimental data;
- industrial/OIPT hardware evidence;
- differential evolution;
- SHGO.

## Next step

Phase 6D may now benchmark the predeclared continuous optimisation candidates
against this deterministic grid baseline.

The deterministic grid remains the mandatory reference. A continuous method
must demonstrate reproducibility, domain compliance, correct infeasibility
handling and scientific consistency with this baseline before it can be
selected.
