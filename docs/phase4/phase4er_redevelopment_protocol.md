# Phase 4E-R — Pre-Test Physics-Constrained Surrogate Redevelopment Protocol

**Status:** PREDECLARED — NO PHASE 4E-R MODEL RESULTS INSPECTED
**Iteration type:** Separate pre-test surrogate-development iteration
**Parent outcome:** Phase 4E CONTROLLED STOP
**Phase 4F status:** LOCKED
**TEST targets:** MUST REMAIN UNAVAILABLE

## 1. Purpose

Phase 4E-R is a new and separate surrogate-development iteration initiated
after the formally closed Phase 4E controlled STOP.

Its purpose is to address the known structural limitation of the Phase 4D
electron-density surrogate without reopening or modifying the completed
Phase 4D/4E experimental record.

Phase 4E-R is not Phase 4F.

Phase 4F remains reserved exclusively for the one-time locked TEST evaluation
after a valid pre-test final surrogate configuration has been frozen.

## 2. TEST lock

Throughout Phase 4E-R:

- TEST target values must remain unavailable;
- `include_test_targets=True` must not be used;
- TEST metrics must not be calculated;
- TEST must not influence model selection;
- TEST must not influence hyperparameter selection;
- TEST must not influence transformation selection;
- TEST must not influence physics-aware acceptance;
- TEST must not influence stopping criteria.

TEST features may only be used where already permitted by the frozen Phase 4
protocol for schema or split-membership verification.

If TEST targets are accidentally accessed, Phase 4E-R must stop and the event
must be documented.

## 3. Frozen dataset and feature contract

Source dataset:

`data/synthetic/phase3/base_steady_state.csv`

Frozen dataset SHA-256:

`dc8d3a24f84b3eca9ce930a88e969e51892d69430ce5840cc546523e9f9dd423`

Frozen split sizes:

- TRAIN: 4,096 rows;
- VALIDATION: 2,048 rows;
- TEST: 2,048 rows.

Approved features remain exactly:

1. `nominal_absorbed_power_W`
2. `target_pressure_mTorr`

No additional feature is permitted.

The qualified numerical envelope remains unchanged:

- absorbed power: 15–90 W;
- pressure: 10–60 mTorr.

## 4. Target contract

Targets remain exactly:

- `true_electron_density_m3`;
- `true_electron_temperature_eV`.

No target definition changes are permitted.

## 5. Scope of redevelopment

Only the electron-density surrogate is redeveloped.

The previously selected electron-temperature configuration is not retuned or
reselected.

The frozen temperature configuration remains:

- estimator: `ExtraTreesRegressor`;
- target transform: identity;
- `n_estimators = 500`;
- `max_depth = None`;
- `min_samples_leaf = 1`;
- `max_features = 1.0`;
- `random_state = 20260913`;
- `n_jobs = 1`.

It must still pass the Phase 4E-R physics-aware acceptance gate together with
the newly selected density candidate.

## 6. Density redevelopment strategy

The density redevelopment introduces an explicitly monotonic constrained
surrogate.

Estimator family:

`HistGradientBoostingRegressor`

Target transformation:

`log10`

The density target transformation is fixed to `log10` for this recovery
iteration.

It will not be reselected.

Because `10**x` is strictly increasing, monotonicity imposed on the transformed
prediction is preserved after inverse transformation to the physical density
scale.

## 7. Density monotonic constraints

Frozen feature order:

1. absorbed power;
2. pressure.

Frozen monotonic constraint:

`monotonic_cst = [1, 1]`

Therefore the fitted density surrogate must be non-decreasing with:

- absorbed power at fixed pressure;
- pressure at fixed absorbed power.

These constraints are chosen to reproduce the measured structural behaviour of
the frozen reduced-order source simulator on the Phase 4E source-reference grid.

They are not claimed to be universal experimental ICP laws.

## 8. Frozen density hyperparameter search

Phase 4E-R reuses the original bounded Phase 4A Histogram Gradient Boosting
search values.

Permitted values:

`learning_rate`:

- 0.05
- 0.10

`max_iter`:

- 200
- 400

`max_leaf_nodes`:

- 15
- 31

`l2_regularization`:

- 0.0
- 0.1

Fixed:

- `monotonic_cst = [1, 1]`
- `early_stopping = False`
- `random_state = 20260913`

Explicitly disabling internal early stopping ensures that every Phase 4E-R
candidate is fitted using the complete 4,096-row TRAIN split and does not create
an estimator-internal validation holdout.

All remaining estimator parameters stay at the scikit-learn 1.9.1 defaults
unless a software-compatibility defect is discovered before any Phase 4E-R
result is evaluated.

Total density candidate configurations:

16.

No Extra Trees retuning is permitted.

No additional estimator family is permitted in this iteration.

No automated or adaptive hyperparameter search is permitted.

## 9. Development data use

For all Phase 4E-R density candidates:

- fit using TRAIN only;
- use exactly 4,096 TRAIN rows;
- evaluate candidates using VALIDATION only;
- use exactly 2,048 VALIDATION rows;
- do not use TEST targets.

Any learned preprocessing must be fitted using TRAIN only.

## 10. Density validation metrics

The complete frozen density metric set remains:

- MAE;
- RMSE;
- R-squared;
- mean absolute relative error;
- median absolute relative error;
- P95 absolute relative error;
- maximum absolute relative error.

All metrics must be reported on the physical electron-density scale after
inverse transformation.

## 11. Density selection rule

The Phase 4A/4D density selection policy remains unchanged.

Primary discriminator:

- P95 absolute relative error.

Candidates within strictly less than 2% relative to the best primary metric
remain in the practical-equivalence cohort.

Secondary discriminators, in order:

1. median absolute relative error;
2. RMSE on the physical density scale;
3. deterministic reproducibility ordering if the declared metrics remain tied.

Candidates outside the strict 2% primary-equivalence band may not re-enter
through secondary metrics.

No new statistical acceptance threshold is introduced.

## 12. Physics-aware acceptance

After validation selection, exactly one density candidate proceeds to
physics-aware acceptance.

The frozen Phase 4E source-reference artifact remains:

`results/phase4/source_reference_grid.json`

Frozen source-reference numerical-array SHA-256:

`4c6af7001e869f35572d451e293c92a1950f1a32c10a29134401a3845380d77a`

The physics-validation design remains exactly:

- 41 absorbed-power values;
- 41 pressure values;
- 1,681 Cartesian probe points;
- absorbed power 15–90 W inclusive;
- pressure 10–60 mTorr inclusive.

The Phase 4E acceptance semantics remain unchanged.

Required checks include:

- finite predictions;
- strictly positive density predictions;
- strictly positive temperature predictions;
- zero source-unsupported trend reversals;
- zero source-unsupported strict local turning points.

The existing Phase 4E acceptance amendment remains authoritative.

No tolerance is introduced.

No probe-grid change is permitted.

No qualified-envelope change is permitted.

## 13. Overall acceptance rule

Both targets must pass.

Phase 4E-R passes only if:

- redeveloped density candidate: PASS;
- frozen temperature candidate: PASS.

If either target fails:

- overall result is STOP;
- no final TRAIN + VALIDATION refit is permitted;
- Phase 4F remains locked;
- no TEST target may be accessed.

## 14. Final TRAIN + VALIDATION refit gate

Only after both targets pass physics-aware acceptance:

- freeze the selected density configuration;
- freeze the existing temperature configuration;
- combine TRAIN + VALIDATION;
- use exactly 6,144 rows;
- rebuild both estimators from their frozen configurations;
- perform no retuning;
- perform no model reselection;
- keep TEST targets unavailable.

The resulting fitted models become the locked pre-test final surrogate
candidates.

They are not yet TEST-evaluated production models.

## 15. Phase 4F unlock condition

Phase 4F may begin only after:

1. Phase 4E-R protocol is frozen in Git;
2. density validation selection is complete;
3. both targets pass physics-aware acceptance;
4. final configurations are frozen;
5. TRAIN + VALIDATION refit uses exactly 6,144 rows;
6. TEST targets remain unaccessed;
7. relevant automated tests pass;
8. full regression passes;
9. machine-readable Phase 4E-R result artifacts are frozen;
10. a clean formal Git checkpoint exists.

Only then may the one-time Phase 4F TEST evaluation occur.

## 16. Failure policy

If Phase 4E-R fails:

- do not weaken the monotonicity requirement;
- do not change the 2% rule;
- do not change the probe grid;
- do not alter the qualified envelope;
- do not inspect TEST;
- do not choose a failed candidate merely because its statistical validation
  error is attractive;
- do not silently start another modelling search.

Any further attempt must again be treated as a separately predeclared
experimental iteration.

## 17. Scientific claim boundary

All existing Phase 4 scientific limitations remain in force:

- synthetic data;
- reduced-order argon plasma model;
- numerically qualified model envelope;
- surrogate of the reduced-order simulator;
- no experimental validation;
- no industrial validation;
- no OIPT operating-range claim;
- absorbed power is not generator RF power;
- no reactive etch/deposition prediction;
- no wafer-scale spatial modelling.

Phase 4E-R demonstrates transferable research-engineering methodology for
physics-aware surrogate development.

It does not reproduce Oxford Instruments Plasma Technology's industrial
process-modelling system.

## 18. Pre-execution declaration

This document must be reviewed and committed before any Phase 4E-R candidate
training, validation evaluation, physics-aware evaluation, or result artifact
generation is performed.

At the time this protocol is frozen:

- no Phase 4E-R candidate result has been inspected;
- TEST targets remain unopened;
- Phase 4F remains locked.
