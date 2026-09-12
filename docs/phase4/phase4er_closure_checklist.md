# Phase 4E-R Closure Checklist

**Closure type:** SUCCESSFUL PRE-TEST REDEVELOPMENT
**Physics-aware acceptance:** PASS
**Final TRAIN + VALIDATION refit:** COMPLETE
**Phase 4F:** UNLOCKED FOR ONE-TIME LOCKED TEST EVALUATION

## Predeclared redevelopment protocol

- [x] Treat Phase 4E-R as a separate pre-test development iteration.
- [x] Freeze the Phase 4E-R protocol before inspecting any new model result.
- [x] Preserve the completed Phase 4D experimental record.
- [x] Preserve the completed Phase 4E controlled-STOP record.
- [x] Preserve the frozen Phase 4 dataset.
- [x] Preserve the frozen two-feature input contract.
- [x] Preserve the frozen two-target output contract.
- [x] Preserve TRAIN = 4,096 rows.
- [x] Preserve VALIDATION = 2,048 rows.
- [x] Preserve TEST = 2,048 rows.
- [x] Keep TEST targets unavailable throughout development.

Protocol:

`docs/phase4/phase4er_redevelopment_protocol.md`

Protocol freeze commit:

`b1a46c0`

Frozen source dataset SHA-256:

`dc8d3a24f84b3eca9ce930a88e969e51892d69430ce5840cc546523e9f9dd423`

## Density redevelopment

- [x] Redevelop only `true_electron_density_m3`.
- [x] Preserve the density `log10` transformation.
- [x] Use `HistGradientBoostingRegressor`.
- [x] Freeze the candidate grid before execution.
- [x] Evaluate exactly 16 density candidates.
- [x] Preserve the original frozen HGBR search dimensions.
- [x] Freeze `monotonic_cst = [1, 1]`.
- [x] Freeze `early_stopping = False`.
- [x] Preserve `random_state = 20260913`.
- [x] Do not retune the failed Phase 4D Extra Trees density model.
- [x] Do not add an adaptive post-result search.
- [x] Do not introduce another model family after seeing results.

Selected density candidate:

`phase4er_hist_gradient_boosting_lr0.05_iter400_leaves31_l20.1`

Selected configuration:

- `learning_rate = 0.05`
- `max_iter = 400`
- `max_leaf_nodes = 31`
- `l2_regularization = 0.1`
- `monotonic_cst = [1, 1]`
- `early_stopping = False`
- `random_state = 20260913`

## Validation selection

- [x] Fit density candidates using TRAIN only.
- [x] Use exactly 4,096 TRAIN rows.
- [x] Evaluate using VALIDATION only.
- [x] Use exactly 2,048 VALIDATION rows.
- [x] Preserve the frozen density validation metric set.
- [x] Preserve P95 absolute relative error as the primary metric.
- [x] Preserve the strict 2% practical-equivalence rule.
- [x] Preserve median absolute relative error as first secondary metric.
- [x] Preserve physical-scale RMSE as second secondary metric.
- [x] Use deterministic ordering only for an exact remaining tie.
- [x] Do not access TEST targets during selection.

Selected validation P95 absolute relative error:

`0.037182267977943516`

Selected validation median absolute relative error:

`0.009727623078162334`

Selected validation RMSE:

`1.6417744711294330e+15 m^-3`

Validation-selection artifact:

`results/phase4/phase4er_validation_selection.json`

Artifact SHA-256:

`453a30cfc1dcd9aa8a9ec561f6de42ba07da4278d242d145863e16489002f388`

Artifact freeze commit:

`f522801`

## Retained temperature surrogate

- [x] Do not redevelop electron temperature.
- [x] Preserve the Phase 4D-selected temperature candidate.
- [x] Preserve identity target transformation.
- [x] Preserve the exact Extra Trees configuration.
- [x] Require the retained temperature model to re-pass physics acceptance.

Temperature candidate:

`extra_trees_n500_depthnone_leaf1_features1`

## Frozen physics-reference contract

- [x] Reuse the exact frozen Phase 4E source-reference grid.
- [x] Preserve the 41 x 41 Cartesian grid.
- [x] Preserve exactly 1,681 probe points.
- [x] Preserve the qualified numerical envelope.
- [x] Preserve the source-reference numerical-array hash.
- [x] Do not add probe points to TRAIN, VALIDATION, or TEST.
- [x] Preserve zero allowed unsupported trend reversals.
- [x] Preserve zero allowed source-unsupported strict turning points.
- [x] Preserve strict positivity requirements.
- [x] Do not alter acceptance tolerances after observing results.

Source-reference numerical-array SHA-256:

`4c6af7001e869f35572d451e293c92a1950f1a32c10a29134401a3845380d77a`

## Physics-aware acceptance

- [x] Rebuild both candidates from frozen configurations.
- [x] Fit acceptance candidates using TRAIN only.
- [x] Use exactly 4,096 TRAIN rows.
- [x] Do not use VALIDATION for acceptance fitting.
- [x] Do not use VALIDATION for acceptance decisions.
- [x] Check density prediction finiteness and strict positivity.
- [x] Check temperature prediction finiteness and strict positivity.
- [x] Check density versus absorbed-power source-referenced trend.
- [x] Check density versus pressure source-referenced trend.
- [x] Check temperature versus pressure source-referenced trend.
- [x] Check density absorbed-power local oscillation.
- [x] Check density pressure local oscillation.
- [x] Check temperature pressure local oscillation.
- [x] Apply the frozen zero-violation acceptance rule.
- [x] Do not access TEST targets.

Density result:

**PASS**

Density observed violations:

- unsupported absorbed-power reversals: `0`;
- unsupported pressure reversals: `0`;
- source-unsupported absorbed-power turning points: `0`;
- source-unsupported pressure turning points: `0`.

Temperature result:

**PASS**

Temperature observed violations:

- unsupported pressure reversals: `0`;
- source-unsupported pressure turning points: `0`.

Overall Phase 4E-R physics-aware acceptance:

**PASS**

Acceptance artifact:

`results/phase4/phase4er_physics_acceptance.json`

Acceptance artifact SHA-256:

`618cabc5fc28be08d24b73c747262b40ed2a6134820098ae0eb33b2d1fc71875`

Acceptance artifact freeze commit:

`4eef0b5`

## Final TRAIN + VALIDATION refit

- [x] Permit final refit only after both targets passed acceptance.
- [x] Freeze the selected density configuration before refit.
- [x] Preserve the frozen temperature configuration.
- [x] Combine TRAIN + VALIDATION only.
- [x] Use exactly 6,144 rows.
- [x] Rebuild both estimators from frozen configurations.
- [x] Perform no retuning.
- [x] Perform no model reselection.
- [x] Perform no transform reselection.
- [x] Preserve density monotonic constraints.
- [x] Preserve density `early_stopping = False`.
- [x] Keep TEST targets unavailable.
- [x] Do not perform TEST evaluation.

Final density refit rows:

`6,144`

Final temperature refit rows:

`6,144`

Final-refit artifact:

`results/phase4/phase4er_final_refit.json`

Final-refit artifact SHA-256:

`4132ff5221c14cf6753e1a19bed5ecaa17577462fcecce1699b57c719a81aa12`

Final-refit artifact freeze commit:

`86f52ec`

## TEST lock

- [x] Default dataset loading withheld TEST targets.
- [x] `include_test_targets=True` was not used.
- [x] TEST metrics were not calculated.
- [x] TEST did not affect candidate selection.
- [x] TEST did not affect hyperparameter selection.
- [x] TEST did not affect transform selection.
- [x] TEST did not affect physics-aware acceptance.
- [x] TEST did not affect the final-refit decision.
- [x] TEST evaluation has not been performed.
- [x] TEST remains uncontaminated for the one-time Phase 4F evaluation.

## Automated verification

- [x] Phase 4E-R targeted regression completed.
- [x] Phase 4E-R targeted regression passed: 49 passed.
- [x] Complete repository regression completed.
- [x] Complete repository regression passed: 359 passed.

## Scientific claim discipline

- [x] Describe the data as synthetic.
- [x] Describe the plasma source model as reduced order.
- [x] Describe the domain as a numerically qualified model envelope.
- [x] Describe the surrogate as a surrogate of the reduced-order simulator.
- [x] Do not claim experimental validation.
- [x] Do not claim industrial validation.
- [x] Do not claim an OIPT operating range.
- [x] Preserve that absorbed power is not generator RF power.
- [x] Do not claim reactive etch or deposition prediction.
- [x] Do not claim wafer-scale spatial modelling.
- [x] Do not describe the frozen physics probe grid as OOD validation.
- [x] Do not describe the monotonic density constraint as a universal ICP law.

## Formal closure gate still required

- [x] Create Phase 4E-R technical summary.
- [x] Create Phase 4E-R closure checklist.
- [x] Run factual QA over Phase 4E-R closure documentation.
- [x] Confirm closure documentation matches frozen artifacts.

- Documentation factual QA: `50 / 50 checks passed`.

- [x] Run staged whitespace check.
- [x] Commit Phase 4E-R closure documentation.
- [x] Confirm clean working tree.
- [x] Formally declare Phase 4E-R closed.
- [x] Formally unlock Phase 4F for the one-time TEST evaluation.

## Phase boundary

Phase 4E-R is formally closed.

Phase 4F is now unlocked exclusively for the one-time locked TEST evaluation
using the frozen pre-test final surrogate configurations.

At this boundary:

- TEST targets have not been accessed;
- TEST metrics have not been calculated;
- the one-time Phase 4F TEST evaluation has not yet been performed;
- no further development decisions are permitted from TEST information.

No further model selection, retuning, hyperparameter modification, transform
change, feature change, acceptance-rule modification, or development iteration
may use TEST information.

Phase 4F TEST results must be treated as final evaluation evidence rather than
development feedback.

## Formal closure record

Phase 4E-R is formally closed as a **successful pre-test redevelopment
iteration**.

Closure evidence:

- predeclared Phase 4E-R protocol frozen before results;
- selected density configuration frozen;
- density physics-aware acceptance: PASS;
- temperature physics-aware acceptance: PASS;
- overall physics-aware acceptance: PASS;
- final TRAIN + VALIDATION refit: COMPLETE;
- final refit rows: 6,144;
- TEST targets: NOT ACCESSED;
- TEST evaluation: NOT PERFORMED;
- targeted Phase 4E-R regression: 49 passed;
- complete repository regression: 359 passed;
- documentation factual QA: 50 / 50 checks passed;
- staged whitespace gate: PASS;
- closure documentation committed at `4ce7447`;
- clean working tree confirmed before formal closure.

The locked pre-test surrogate configuration is therefore eligible to enter
Phase 4F.

Phase 4F is restricted to the one-time final TEST evaluation.
