# Phase 4F — Locked TEST Evaluation Execution Protocol

## Status

**Protocol status:** FROZEN BEFORE TEST TARGET ACCESS
**Phase:** 4F
**Purpose:** One-time locked TEST evaluation of the formally frozen Phase 4E-R surrogate configuration.

Phase 4E-R was formally closed at Git commit:

`0f604d7`

At the time this protocol is frozen:

- TEST targets have not been accessed;
- TEST metrics have not been calculated;
- no Phase 4F TEST evaluation has been performed;
- final surrogate configurations are frozen;
- no further model selection or retuning is permitted.

---

## 1. Frozen data-use contract

Dataset:

`data/synthetic/phase3/base_steady_state.csv`

Frozen split sizes:

- TRAIN: 4,096 rows
- VALIDATION: 2,048 rows
- TEST: 2,048 rows

The final surrogate is fitted on exactly:

`TRAIN + VALIDATION = 6,144 rows`

The TEST split is used once for final generalisation evaluation.

TEST targets may be exposed only by the dedicated Phase 4F evaluation path using:

`load_phase4_dataset(include_test_targets=True)`

No other Phase 4 development code may access TEST targets.

---

## 2. Frozen features

Exact feature order:

1. `nominal_absorbed_power_W`
2. `target_pressure_mTorr`

No additional features are permitted.

---

## 3. Frozen final density surrogate

Target:

`true_electron_density_m3`

Model:

`HistGradientBoostingRegressor`

Target transform:

`log10`

Frozen parameters:

- `learning_rate = 0.05`
- `max_iter = 400`
- `max_leaf_nodes = 31`
- `l2_regularization = 0.1`
- `monotonic_cst = [1, 1]`
- `early_stopping = False`
- `random_state = 20260913`

The density surrogate must be reconstructed through the existing frozen
Phase 4E-R final-refit implementation.

---

## 4. Frozen final temperature surrogate

Target:

`true_electron_temperature_eV`

Model:

`ExtraTreesRegressor`

Target transform:

`identity`

Frozen parameters:

- `n_estimators = 500`
- `max_depth = None`
- `min_samples_leaf = 1`
- `max_features = 1.0`
- `random_state = 20260913`

The temperature surrogate must be reconstructed through the existing frozen
Phase 4E-R final-refit implementation.

---

## 5. Full TEST metric reporting

The existing Phase 4 metric implementation must be reused directly.

### Electron density

Report:

- MAE
- RMSE
- R²
- mean absolute relative error
- median absolute relative error
- P95 absolute relative error
- maximum absolute relative error

### Electron temperature

Report:

- MAE in eV
- RMSE in eV
- R²
- mean absolute relative error
- median absolute relative error
- P95 absolute error in eV
- maximum absolute error in eV

No new TEST-driven metric may replace or suppress these metrics.

---

## 6. Physical-domain normalisation

Frozen physical domain:

- absorbed power: 15–90 W
- pressure: 10–60 mTorr

Normalized coordinates:

`P_norm = (P - 15) / (90 - 15)`

`p_norm = (p - 10) / (60 - 10)`

The physical limits, not empirical TEST quantiles, define all regions.

---

## 7. Boundary and interior summaries

A TEST point is **near-boundary** when either normalized coordinate is:

- `<= 0.10`, or
- `>= 0.90`

A TEST point is **interior** only when both normalized coordinates are:

- `> 0.10`, and
- `< 0.90`

These Phase 4 definitions must be recomputed directly from the frozen physical
coordinates.

The Phase 3 `domain_status` column must not be reused for this purpose because
its near-boundary threshold belongs to the Phase 3 dataset-generation contract.

For each target, report the full frozen target-specific metric set separately
for:

- near-boundary TEST points;
- interior TEST points.

Also report row counts.

---

## 8. Central-domain diagnostic

A point belongs to the central-domain diagnostic region when both normalized
coordinates lie within:

`[0.20, 0.80]`

The interval is inclusive.

For each target report:

- row count;
- full frozen target-specific metric set.

---

## 9. Operating-space thirds

For either normalized coordinate:

### Low

`0.0 <= x_norm < 1/3`

### Middle

`1/3 <= x_norm < 2/3`

### High

`2/3 <= x_norm <= 1.0`

For each target report the full frozen metric set for:

### Power regions

- low power;
- middle power;
- high power.

### Pressure regions

- low pressure;
- middle pressure;
- high pressure.

Each summary must include its row count.

---

## 10. Corner diagnostics

Using the frozen low/high thirds, report target-specific metrics for:

- low power / low pressure;
- low power / high pressure;
- high power / low pressure;
- high power / high pressure.

Each corner summary must include its row count.

---

## 11. Worst-case TEST observations

Exactly **10 worst observations per target** will be reported.

This number is frozen before TEST is opened.

### Density ranking

Rank by:

`absolute relative error`

descending.

### Temperature ranking

Rank by:

`absolute error in eV`

descending.

### Deterministic tie-break

If two observations have exactly equal ranking error, sort by original TEST
split index ascending.

Each reported worst-case observation will include:

- TEST split index;
- nominal absorbed power in W;
- target pressure in mTorr;
- true target value;
- predicted target value;
- absolute error;
- absolute relative error.

Worst-case observations are descriptive final-evaluation evidence only.

They must not trigger model changes.

---

## 12. Prediction-integrity checks

For both targets report:

- prediction count;
- all predictions finite;
- minimum predicted value;
- maximum predicted value;
- all predictions strictly positive.

A prediction-integrity defect must be reported honestly.

It must not trigger retuning against the same TEST set.

---

## 13. Structural diagnostics

The final 6,144-row-refitted surrogate may be evaluated on the already frozen
41 x 41 Phase 4 physics probe grid.

This diagnostic does not add training data and does not use TEST target values.

The existing frozen source-reference behaviour and Phase 4E-R structural rules
must be reused.

Structural diagnostics are reporting evidence after final refit, not a new
model-selection gate.

Any unexpected structural defect must be reported and must not trigger Phase 4
retuning from TEST information.

---

## 14. TEST interpretation

No TEST-performance threshold is being introduced in Phase 4F.

Phase 4F reports final generalisation evidence.

Poor or unexpected TEST performance must not be hidden and must not result in:

- model reselection;
- hyperparameter modification;
- transformation modification;
- feature modification;
- acceptance-rule modification;
- TEST-informed redevelopment.

If a major defect is discovered, it must be documented as such. The same TEST
set must not subsequently become development or validation data.

---

## 15. Result artifact

The one-time locked evaluation will be written to:

`results/phase4/locked_test_evaluation.json`

The writer must refuse to overwrite an existing result artifact unless a
separate explicitly documented recovery procedure is created.

The artifact will record:

- frozen configuration identity;
- fit-row count;
- TEST-row count;
- TEST metrics;
- error-space summaries;
- worst-case observations;
- prediction-integrity diagnostics;
- structural diagnostics;
- TEST access state;
- scientific-scope metadata.

---

## 16. Scientific claim boundary

All Phase 4F reporting must preserve that:

- the data are synthetic;
- the plasma source is a reduced-order argon plasma model;
- the domain is a numerically qualified model envelope;
- the surrogate approximates the reduced-order simulator;
- there is no experimental validation;
- there is no industrial validation;
- there is no OIPT operating-range claim;
- absorbed power is not generator RF power;
- reactive etch/deposition behaviour is not predicted;
- wafer-scale spatial behaviour is not modelled.

TEST performance is therefore synthetic simulator-surrogate generalisation
evidence, not experimental or industrial process-validation evidence.

---

## 17. Irreversibility rule

Once TEST targets are opened for the real Phase 4F run:

- the TEST evaluation is considered consumed;
- the resulting metrics are final Phase 4 evidence;
- no further Phase 4 surrogate development may use those TEST results.

Therefore implementation and automated tests for Phase 4F must be completed
before the real TEST-target access occurs.
