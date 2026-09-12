# Phase 4E-R - Pre-Test Physics-Constrained Surrogate Redevelopment

**Status:** FORMALLY CLOSED - PRE-TEST REDEVELOPMENT PASSED
**Physics-aware acceptance:** PASS
**Final TRAIN + VALIDATION refit:** COMPLETE
**Phase 4F status:** UNLOCKED FOR ONE-TIME LOCKED TEST EVALUATION

## 1. Purpose

Phase 4E-R was a separately predeclared pre-test surrogate-development
iteration initiated after the formally closed Phase 4E controlled STOP.

Phase 4E had stopped because the Phase 4D-selected electron-density
Extra Trees surrogate produced one source-unsupported pressure-direction
reversal and one corresponding source-unsupported strict turning point on the
frozen 41 x 41 physics-validation grid.

Phase 4E-R did not reopen or alter the completed Phase 4D or Phase 4E
experimental record.

Instead, before any new model result was inspected, Phase 4E-R froze a separate
redevelopment protocol that:

- preserved the original Phase 4 dataset;
- preserved the two frozen input features;
- preserved the two physical targets;
- preserved the TRAIN / VALIDATION / TEST split;
- preserved the locked TEST policy;
- preserved the Phase 4 validation metric definitions;
- preserved the strict 2% practical-equivalence rule;
- preserved the original frozen physics-acceptance grid;
- preserved the original zero-violation physics-acceptance rules;
- redeveloped only the electron-density surrogate;
- retained the existing Phase 4D temperature surrogate;
- introduced a source-structure-informed monotonic constraint for density;
- prohibited post-result retuning or acceptance-rule modification.

Phase 4E-R is not Phase 4F.

TEST target values remained unavailable throughout Phase 4E-R.

## 2. Frozen Phase 4E-R protocol

Protocol document:

`docs/phase4/phase4er_redevelopment_protocol.md`

Protocol freeze commit:

`b1a46c0`

The protocol was committed before any Phase 4E-R model result was generated.

The frozen data contract remained:

- TRAIN: 4,096 rows;
- VALIDATION: 2,048 rows;
- TEST: 2,048 rows.

Frozen feature order:

1. `nominal_absorbed_power_W`
2. `target_pressure_mTorr`

Frozen targets:

1. `true_electron_density_m3`
2. `true_electron_temperature_eV`

The source dataset remained:

`data/synthetic/phase3/base_steady_state.csv`

Frozen dataset SHA-256:

`dc8d3a24f84b3eca9ce930a88e969e51892d69430ce5840cc546523e9f9dd423`

TEST targets were not accessed.

`include_test_targets=True` was not used.

## 3. Density redevelopment design

Only the electron-density surrogate was redeveloped.

Target:

`true_electron_density_m3`

Frozen target transformation:

`log10`

Model family:

`HistGradientBoostingRegressor`

The density redevelopment reused the previously frozen Phase 4
HistGradientBoosting search dimensions:

- `learning_rate`: `0.05`, `0.10`;
- `max_iter`: `200`, `400`;
- `max_leaf_nodes`: `15`, `31`;
- `l2_regularization`: `0.0`, `0.1`.

Total frozen candidate configurations:

`16`

Additional Phase 4E-R fixed settings:

- `monotonic_cst = [1, 1]`;
- `early_stopping = False`;
- `random_state = 20260913`.

The monotonic constraint means the fitted density surrogate is constrained to
be non-decreasing with:

- absorbed power;
- pressure.

This constraint was motivated by the measured behaviour of the frozen
reduced-order source simulator inside the qualified model envelope.

It is not asserted as a universal experimental ICP law.

No Extra Trees retuning was performed.

No additional model family was introduced.

No adaptive or post-result hyperparameter search was performed.

## 4. Validation selection

Each of the 16 density candidates was:

- fitted using TRAIN only;
- fitted on exactly 4,096 TRAIN rows;
- evaluated using VALIDATION only;
- evaluated on exactly 2,048 VALIDATION rows;
- evaluated on the physical density scale after inverse transformation.

TEST targets were unavailable.

The frozen density selection hierarchy remained:

1. primary metric:
   `p95_absolute_relative_error`;
2. strict practical-equivalence cohort:
   candidate primary metric must be strictly less than
   `best_primary * 1.02`;
3. first secondary metric:
   `median_absolute_relative_error`;
4. second secondary metric:
   physical-scale `rmse`;
5. deterministic frozen candidate ordering only if an exact tie remained.

The selected density candidate was:

`phase4er_hist_gradient_boosting_lr0.05_iter400_leaves31_l20.1`

Selected configuration:

- model:
  `HistGradientBoostingRegressor`;
- transformation:
  `log10`;
- `learning_rate = 0.05`;
- `max_iter = 400`;
- `max_leaf_nodes = 31`;
- `l2_regularization = 0.1`;
- `monotonic_cst = [1, 1]`;
- `early_stopping = False`;
- `random_state = 20260913`.

Best observed primary metric:

`0.03707370717580944`

Strict primary-equivalence upper boundary:

`0.03781518131932563`

The selected candidate belonged to the frozen primary-equivalence cohort and
was selected at the first secondary criterion:

`median_absolute_relative_error`

Selected validation metrics:

- MAE:
  `1.2137336212248678e+15 m^-3`;
- RMSE:
  `1.6417744711294330e+15 m^-3`;
- R2:
  `0.9988060602389571`;
- mean absolute relative error:
  `0.013643283653820577`;
- median absolute relative error:
  `0.009727623078162334`;
- P95 absolute relative error:
  `0.037182267977943516`;
- maximum absolute relative error:
  `0.08391224389007498`.

The P95 relative validation error is therefore approximately:

`3.7182%`

This value is a validation-selection result.

It is not a TEST result.

Frozen validation-selection artifact:

`results/phase4/phase4er_validation_selection.json`

Artifact SHA-256:

`453a30cfc1dcd9aa8a9ec561f6de42ba07da4278d242d145863e16489002f388`

Artifact freeze commit:

`f522801`

## 5. Retained temperature configuration

The electron-temperature surrogate was not redeveloped.

Target:

`true_electron_temperature_eV`

The exact Phase 4D-selected configuration was retained:

- model:
  `ExtraTreesRegressor`;
- transformation:
  `identity`;
- candidate ID:
  `extra_trees_n500_depthnone_leaf1_features1`;
- `n_estimators = 500`;
- `max_depth = None`;
- `min_samples_leaf = 1`;
- `max_features = 1.0`;
- `random_state = 20260913`;
- `n_jobs = 1`.

The temperature candidate was required to pass the frozen physics-aware
acceptance gate again alongside the redeveloped density candidate.

## 6. Frozen physics-reference contract

Phase 4E-R reused exactly the existing frozen Phase 4E source-reference grid.

No new source grid was generated.

Probe-grid design:

- absorbed power:
  41 equally spaced values from 15 W to 90 W inclusive;
- pressure:
  41 equally spaced values from 10 mTorr to 60 mTorr inclusive;
- total points:
  1,681;
- unique points:
  1,681.

Frozen source-reference artifact:

`results/phase4/source_reference_grid.json`

Frozen numerical-array SHA-256:

`4c6af7001e869f35572d451e293c92a1950f1a32c10a29134401a3845380d77a`

The same zero-tolerance acceptance semantics were retained:

- every surrogate prediction must be finite;
- every surrogate prediction must be strictly positive;
- zero source-unsupported adjacent trend reversals are allowed;
- zero source-unsupported strict local turning points are allowed.

No tolerance was changed after seeing the Phase 4E-R results.

## 7. Phase 4E-R physics-aware acceptance

For physics-aware acceptance, both candidates were rebuilt from their frozen
configurations and fitted using TRAIN only.

Fit rows:

`4,096`

VALIDATION was not used for acceptance fitting.

TEST targets were not accessed.

### Density acceptance

Candidate:

`phase4er_hist_gradient_boosting_lr0.05_iter400_leaves31_l20.1`

Physical-scale probe prediction range:

- minimum:
  `1.668951770588742e+16 m^-3`;
- maximum:
  `2.2740522043077328e+17 m^-3`.

Positivity:

- non-finite predictions:
  `0`;
- non-positive predictions:
  `0`;
- result:
  **PASS**.

Density versus absorbed power:

- unsupported source-referenced trend reversals:
  `0`;
- source-unsupported strict turning points:
  `0`;
- result:
  **PASS**.

Density versus pressure:

- unsupported source-referenced trend reversals:
  `0`;
- source-unsupported strict turning points:
  `0`;
- result:
  **PASS**.

Density overall acceptance:

**PASS**

The local pressure reversal that caused the original Phase 4E controlled STOP
is not present in the Phase 4E-R accepted density surrogate.

### Temperature acceptance

Candidate:

`extra_trees_n500_depthnone_leaf1_features1`

Physical-scale probe prediction range:

- minimum:
  `1.5473202323272093 eV`;
- maximum:
  `2.074477760192988 eV`.

Positivity:

- non-finite predictions:
  `0`;
- non-positive predictions:
  `0`;
- result:
  **PASS**.

Temperature versus pressure:

- unsupported source-referenced trend reversals:
  `0`;
- source-unsupported strict turning points:
  `0`;
- result:
  **PASS**.

Temperature overall acceptance:

**PASS**

### Overall physics-aware acceptance

Frozen rule:

Both target-specific candidates must pass.

Observed result:

**PASS**

Final TRAIN + VALIDATION refit was therefore permitted.

Frozen acceptance artifact:

`results/phase4/phase4er_physics_acceptance.json`

Acceptance artifact SHA-256:

`618cabc5fc28be08d24b73c747262b40ed2a6134820098ae0eb33b2d1fc71875`

Acceptance artifact freeze commit:

`4eef0b5`

## 8. Final TRAIN + VALIDATION refit

After both targets passed physics-aware acceptance, the two configurations were
frozen and rebuilt for the final pre-test refit.

No hyperparameter was changed.

No transform was changed.

No model was reselected.

No TEST target was accessed.

Fit splits:

- TRAIN;
- VALIDATION.

Fit row counts:

- TRAIN:
  `4,096`;
- VALIDATION:
  `2,048`;
- combined:
  `6,144`.

### Final density configuration

Candidate:

`phase4er_hist_gradient_boosting_lr0.05_iter400_leaves31_l20.1`

Configuration remained:

- transformation:
  `log10`;
- `learning_rate = 0.05`;
- `max_iter = 400`;
- `max_leaf_nodes = 31`;
- `l2_regularization = 0.1`;
- `monotonic_cst = [1, 1]`;
- `early_stopping = False`;
- `random_state = 20260913`.

Final refit rows:

`6,144`

### Final temperature configuration

Candidate:

`extra_trees_n500_depthnone_leaf1_features1`

Configuration remained:

- transformation:
  `identity`;
- `n_estimators = 500`;
- `max_depth = None`;
- `min_samples_leaf = 1`;
- `max_features = 1.0`;
- `random_state = 20260913`;
- `n_jobs = 1`.

Final refit rows:

`6,144`

The resulting fitted estimators are the locked pre-test final surrogate
candidates.

They have not yet undergone the one-time locked TEST evaluation.

Frozen final-refit result artifact:

`results/phase4/phase4er_final_refit.json`

Final-refit artifact SHA-256:

`4132ff5221c14cf6753e1a19bed5ecaa17577462fcecce1699b57c719a81aa12`

Final-refit artifact freeze commit:

`86f52ec`

The artifact records configuration and refit metadata.

It does not claim that a TEST-evaluated production model has been established.

## 9. TEST lock

Throughout Phase 4E-R:

- TEST targets remained unavailable;
- `include_test_targets=True` was not used;
- TEST metrics were not calculated;
- TEST did not influence model selection;
- TEST did not influence transformation selection;
- TEST did not influence hyperparameter selection;
- TEST did not influence physics-aware acceptance;
- TEST did not influence the final-refit decision;
- TEST evaluation was not performed.

Phase 4F therefore remains uncontaminated by development information.

## 10. Automated verification

Phase 4E-R targeted regression:

`49 passed`

Complete repository regression:

`359 passed`

The complete regression suite therefore passes after the Phase 4E-R
implementation and final refit.

## 11. Phase 4E-R outcome

Phase 4E-R successfully resolves the specific pre-test structural limitation
that caused the Phase 4E controlled STOP while preserving the locked TEST set.

Final Phase 4E-R technical outcome:

- density validation selection:
  **COMPLETE**;
- density physics-aware acceptance:
  **PASS**;
- temperature physics-aware acceptance:
  **PASS**;
- overall physics-aware acceptance:
  **PASS**;
- final TRAIN + VALIDATION refit:
  **COMPLETE**;
- final refit rows:
  `6,144`;
- TEST targets accessed:
  **NO**;
- TEST evaluation performed:
  **NO**;
- targeted regression:
  **49 passed**;
- full repository regression:
  **359 passed**.

No post-result acceptance tolerance was introduced.

No failed candidate was silently substituted.

No TEST information was used to guide redevelopment.

## 12. Scientific interpretation and claim boundary

The Phase 4E-R result demonstrates that a predeclared monotonic-constrained
density surrogate can reproduce the relevant measured structural behaviour of
the project's reduced-order source simulator across the frozen qualified
diagnostic grid while retaining strong validation accuracy.

The project remains subject to the following claim boundary:

- the data are synthetic;
- the source plasma model is reduced order;
- the operating domain is a numerically qualified model envelope;
- the surrogate approximates the reduced-order simulator;
- no experimental plasma dataset has validated the surrogate;
- no industrial process validation has been performed;
- no OIPT operating-range claim is made;
- absorbed power is not equivalent to generator RF power;
- no reactive etch or deposition chemistry is predicted;
- no wafer-scale spatial plasma field is modelled.

The Phase 4E-R monotonic density behaviour is a constraint aligned with the
measured structure of this specific reduced-order source model inside the
frozen qualified envelope.

It must not be interpreted as a universal physical law for ICP systems.

## 13. Phase boundary

Phase 4E-R is formally closed.

All frozen pre-test requirements have been satisfied:

- the Phase 4E-R protocol was frozen before redevelopment results;
- density validation selection is complete;
- both targets passed frozen physics-aware acceptance;
- final surrogate configurations are frozen;
- TRAIN + VALIDATION refit used exactly 6,144 rows;
- TEST targets remained unaccessed;
- targeted regression passed: 49 tests;
- full repository regression passed: 359 tests;
- closure documentation factual QA passed: 50 / 50 checks;
- closure documentation was committed;
- a clean Git checkpoint was confirmed before formal closure.

Phase 4F is therefore unlocked exclusively for the one-time locked TEST
evaluation.

No Phase 4F TEST evaluation has yet been performed.

TEST targets remain unopened at the Phase 4E-R boundary.

From this point onward, no model selection, retuning, transform change,
hyperparameter change, feature change, physics-rule change, or development
iteration may use TEST information.

Phase 4F TEST results must be treated as final evaluation evidence and not as
feedback for further surrogate development.
