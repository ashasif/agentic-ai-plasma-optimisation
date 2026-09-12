# Phase 4D - Controlled Validation-Based Surrogate Selection

**Status:** COMPLETE - VALIDATED AND READY FOR CLOSURE GATE

## 1. Scope

Phase 4D performed the formal controlled validation-based surrogate-selection
procedure defined by the frozen Phase 4A modelling protocol.

The phase exhaustively evaluated the permitted classical surrogate search space
using:

- TRAIN only for model fitting;
- VALIDATION only for model, hyperparameter and density-transform selection.

The evaluated classical model families were:

- polynomial response-surface regression;
- ExtraTreesRegressor;
- HistGradientBoostingRegressor.

Electron-density candidates were evaluated under both frozen target
representations:

- identity;
- log10.

Electron-temperature candidates were evaluated using:

- identity only.

Phase 4D selected validation-stage candidates for:

- electron density;
- electron temperature;
- the electron-density target transformation;
- model-family configuration;
- permitted hyperparameters.

TEST target values were not accessed.

No TRAIN + VALIDATION refit was performed.

No Phase 4E physics-aware probe-grid acceptance was performed.

No locked TEST evaluation was performed.

No production surrogate was persisted.

The selected models therefore remain VALIDATION-SELECTED CANDIDATES only.

They must still pass Phase 4E pre-test physics-aware acceptance before the
locked TEST evaluation permitted in Phase 4F.

## 2. Frozen data and modelling contract

Source dataset:

`data/synthetic/phase3/base_steady_state.csv`

Frozen SHA-256:

`dc8d3a24f84b3eca9ce930a88e969e51892d69430ce5840cc546523e9f9dd423`

Rows:

- TRAIN: 4,096
- VALIDATION: 2,048
- TEST: 2,048
- Total: 8,192

Approved input features:

- `nominal_absorbed_power_W`
- `target_pressure_mTorr`

Primary targets:

- `true_electron_density_m3`
- `true_electron_temperature_eV`

The existing Phase 4 controlled loader was reused.

The Phase 4 loader was called without enabling TEST targets.

Observed Phase 4D data use was:

- fitting split: TRAIN;
- evaluation split: VALIDATION;
- TEST rows structurally present: 2,048;
- TEST targets accessed: no.

The monitoring/fault dataset remained excluded.

No solver diagnostics, convergence information, balance variables, monitoring
features or other Phase 3 columns were introduced into the surrogate feature
contract.

## 3. Phase 4D implementation

Phase 4D added:

`src/plasma_ai/surrogate/selection.py`

This module provides:

- deterministic enumeration of the frozen classical search grid;
- exact frozen-grid family counts;
- frozen-grid uniqueness checks;
- strict 2% primary-metric practical-equivalence handling;
- target-specific primary and secondary selection policies;
- deterministic simplicity tie-breaking;
- deterministic canonical fallback ordering;
- transform-qualified selection-trace metadata;
- explicit primary-equivalence boundary metadata.

Phase 4D also added:

`src/plasma_ai/surrogate/selection_experiment.py`

This module provides:

- complete Phase 4D TRAIN/VALIDATION orchestration;
- explicit TEST-lock enforcement;
- all 108 controlled validation fits;
- joint density-transform/model selection;
- temperature-model selection;
- structured machine-readable selection output;
- scientific-scope metadata;
- explicit Phase 4E/4F/4G boundary metadata.

The existing Phase 4C fitting infrastructure was reused:

`src/plasma_ai/surrogate/classical.py`

No duplicate surrogate-fitting implementation was introduced.

The existing fitting path continues to provide:

- target-transform validation;
- TRAIN-only fitting;
- inverse transformation before physical-scale metrics;
- deterministic estimator construction;
- frozen parameter validation;
- numerical prediction-sanity metadata.

## 4. Frozen search-space enumeration

Phase 4D evaluated exactly 36 classical configurations for each approved
target/transform combination.

### 4.1 Polynomial response surfaces

Frozen degrees:

- degree 2;
- degree 3.

Total polynomial configurations:

- 2.

Polynomial expansion includes powers and interaction terms.

The estimator remains ordinary LinearRegression after polynomial feature
expansion.

No Ridge regularisation was introduced.

### 4.2 Extra Trees

Frozen search space:

`n_estimators`:

- 200;
- 500.

`max_depth`:

- None;
- 8;
- 16.

`min_samples_leaf`:

- 1;
- 2;
- 4.

`max_features`:

- 1.0.

Fixed random state:

- `20260913`.

Total Extra Trees configurations:

- 18.

### 4.3 Histogram Gradient Boosting

Frozen search space:

`learning_rate`:

- 0.05;
- 0.10.

`max_iter`:

- 200;
- 400.

`max_leaf_nodes`:

- 15;
- 31.

`l2_regularization`:

- 0.0;
- 0.1.

Fixed random state:

- `20260913`.

Total Histogram Gradient Boosting configurations:

- 16.

### 4.4 Total controlled search

Per target/transform combination:

- polynomial: 2;
- Extra Trees: 18;
- Histogram Gradient Boosting: 16;
- total: 36.

Target/transform combinations:

1. electron density - identity;
2. electron density - log10;
3. electron temperature - identity.

Total controlled TRAIN-to-VALIDATION fits:

- 108.

MLP remained disabled.

No model family or parameter value outside the frozen Phase 4A protocol was
introduced.

## 5. Frozen selection policy

### 5.1 Electron density

Primary validation discriminator:

- P95 absolute relative error.

Secondary discriminators, in frozen order:

1. median absolute relative error;
2. RMSE on physical electron-density scale;
3. simplicity / reproducibility if the declared metrics remain tied.

### 5.2 Electron temperature

Primary validation discriminator:

- RMSE in eV.

Secondary discriminators, in frozen order:

1. P95 absolute error in eV;
2. MAE in eV;
3. simplicity / reproducibility if the declared metrics remain tied.

R-squared was reported but was not used as the primary selector.

No aggregate multi-metric objective was introduced.

## 6. Implementation of the 2% practical-equivalence rule

For each target, the globally best primary error metric defines the reference
score.

For a strictly positive best primary value, a candidate remains within the
practical-equivalence cohort only when its primary metric is strictly less than:

`best_primary_value * (1 + 0.02)`

The boundary is strict because the frozen Phase 4A protocol states that a
primary-metric difference smaller than 2% is practically marginal.

A candidate exactly 2% worse than the best result is therefore outside the
equivalence cohort.

If the best score is exactly zero, only other zero-score candidates are treated
as equivalent.

Candidates outside the primary-equivalence cohort cannot re-enter the selection
through secondary metrics or simplicity.

For candidates inside the equivalence cohort:

1. the first frozen secondary metric is applied;
2. the second frozen secondary metric is applied if required;
3. simplicity / reproducibility is used only if the declared statistical
   metrics remain tied.

No additional 2% threshold was invented for secondary metrics.

## 7. Deterministic simplicity and reproducibility policy

Simplicity is not permitted to override materially better primary validation
performance.

It is considered only after candidates remain indistinguishable under the
frozen primary-equivalence and secondary-metric policy.

The deterministic family hierarchy is:

1. polynomial response surface;
2. Extra Trees;
3. Histogram Gradient Boosting.

Within polynomial regression:

- degree 2 precedes degree 3 when all declared metrics remain tied.

No arbitrary numerical model-complexity score was created.

No artificial complexity ranking was introduced between different Extra Trees
or Histogram Gradient Boosting hyperparameter settings.

If candidates inside the same non-polynomial family remain exactly tied under
the frozen metric policy, the canonical frozen-grid enumeration order is used
only as a final deterministic reproducibility fallback.

## 8. Structured Phase 4D artifact

Structured result artifact:

`results/phase4/validation_selection.json`

Artifact SHA-256 on the current Windows filesystem:

`1d95d8ac7606e8a838920bea488cf4c3818eb7e072129b1e924737638eff85b2`

Artifact size at generation:

- 107,629 bytes.

The artifact contains:

- Phase 4D phase and experiment markers;
- scientific-scope metadata;
- split-use metadata;
- controlled-search design metadata;
- all 108 validation results;
- physical-scale metrics;
- prediction-sanity metadata;
- selected density candidate;
- selected density target transformation;
- selected temperature candidate;
- selected hyperparameters;
- primary-equivalence cohorts;
- best primary scores;
- strict 2% upper boundaries;
- selection stages;
- explicit Phase 4E/4F/4G boundary metadata.

## 9. Electron-density validation selection

The density selection pool contained:

- 36 identity candidates;
- 36 log10 candidates;
- 72 total validation candidates.

Both density target representations were compared in one controlled selection
pool.

The target representation was therefore selected through the same frozen
validation policy as model family and hyperparameters.

### 9.1 Best primary validation result

The lowest observed density P95 absolute relative error was:

`0.0039696471210895435`

or approximately:

- 0.396965%.

It was produced by:

- model: Extra Trees;
- transform: log10;
- `n_estimators = 500`;
- `max_depth = None`;
- `min_samples_leaf = 1`;
- `max_features = 1.0`.

The strict 2% practical-equivalence upper boundary was:

`0.0040490400635113345`

or approximately:

- 0.404904%.

### 9.2 Density primary-equivalence cohort

Three candidates fell inside the frozen strict 2% primary-equivalence band:

1. Extra Trees, 500 estimators, unlimited depth, leaf size 1,
   identity transform;
2. Extra Trees, 500 estimators, unlimited depth, leaf size 1,
   log10 transform;
3. Extra Trees, 500 estimators, depth 16, leaf size 1,
   log10 transform.

A fourth closely performing configuration:

- Extra Trees;
- 500 estimators;
- depth 16;
- leaf size 1;
- identity transform;

had P95 absolute relative error:

`0.004122821612`

and therefore lay outside the strict 2% equivalence boundary.

It was not permitted to re-enter through secondary metrics.

### 9.3 Validation-selected density candidate

The selected density candidate was:

- model: Extra Trees;
- transform: log10;
- `n_estimators = 500`;
- `max_depth = 16`;
- `min_samples_leaf = 1`;
- `max_features = 1.0`;
- `random_state = 20260913`.

Selection stage:

`secondary_metric:median_absolute_relative_error`

The selected candidate did not have the numerically lowest primary metric.

Its P95 absolute relative error remained inside the frozen 2% practical
equivalence band, after which the first predeclared secondary metric selected
the candidate.

Validation metrics for the selected density candidate were:

- MAE: approximately `1.16705e14 m^-3`;
- RMSE: approximately `1.73808e14 m^-3`;
- R-squared: `0.999986619`;
- mean absolute relative error: approximately `0.13836%`;
- median absolute relative error: approximately `0.09135%`;
- P95 absolute relative error: approximately `0.39845%`;
- maximum absolute relative error: approximately `3.52969%`.

This is the Phase 4D validation-selected density candidate.

It is not yet a final accepted or locked-test-evaluated surrogate.

## 10. Electron-temperature validation selection

The electron-temperature selection pool contained:

- 36 identity candidates.

The frozen primary discriminator was RMSE in eV.

### 10.1 Best primary validation result

The lowest observed validation RMSE was:

`1.1002626493505393e-05 eV`

The strict 2% practical-equivalence upper boundary was:

`1.1222679023375502e-05 eV`

Only one candidate lay inside this strict primary-equivalence band.

No secondary metric or simplicity tie-break was therefore required.

### 10.2 Validation-selected temperature candidate

The selected temperature candidate was:

- model: Extra Trees;
- transform: identity;
- `n_estimators = 500`;
- `max_depth = None`;
- `min_samples_leaf = 1`;
- `max_features = 1.0`;
- `random_state = 20260913`.

Selection stage:

`primary_metric`

Validation metrics for the selected temperature candidate were:

- MAE: approximately `6.77764e-06 eV`;
- RMSE: approximately `1.10026e-05 eV`;
- R-squared: approximately `0.9999999934`;
- mean absolute relative error: approximately `0.000377%`;
- median absolute relative error: approximately `0.000223%`;
- P95 absolute error: approximately `2.30877e-05 eV`;
- maximum absolute error: approximately `8.38035e-05 eV`.

This is the Phase 4D validation-selected temperature candidate.

It is not yet a final accepted or locked-test-evaluated surrogate.

## 11. Relationship to the Phase 4C benchmark

Phase 4C used representative points from the frozen classical search spaces and
did not perform model selection.

The Phase 4C Extra Trees reference configuration used:

- `n_estimators = 200`;
- `max_depth = None`;
- `min_samples_leaf = 1`;
- `max_features = 1.0`.

Phase 4D evaluated the complete permitted grids.

For electron density, Phase 4C observed a best representative Extra Trees P95
absolute relative error of approximately:

- 0.4369% for log10.

The Phase 4D selected density candidate achieved approximately:

- 0.39845%.

For electron temperature, the Phase 4C Extra Trees reference configuration
achieved RMSE of approximately:

- `1.32496e-05 eV`.

The Phase 4D selected temperature candidate achieved approximately:

- `1.10026e-05 eV`.

The Phase 4D search therefore improved on the representative Phase 4C Extra
Trees configurations while remaining entirely inside the predeclared Phase 4A
search spaces.

These are VALIDATION comparisons only.

## 12. Numerical sanity checks

All 108 Phase 4D validation results passed the lightweight numerical sanity
checks.

Across all validation predictions:

- all predictions were finite;
- total candidates with negative predictions: 0;
- total candidates with non-positive predictions: 0;
- total candidates with sanity concerns: 0.

No non-finite inverse-transform results were observed.

No unexpected parameter values outside the frozen grids were admitted.

The full physics-aware probe-grid, monotonicity, trend and local-oscillation
assessment remains reserved for Phase 4E.

## 13. Artifact reconstruction audit

The stored Phase 4D artifact was independently reconstructed using the 108
stored candidate results.

The frozen selector was reapplied to:

- all 72 density candidates;
- all 36 temperature candidates.

The reconstructed density decision exactly matched the stored decision:

- candidate ID;
- model family;
- transform;
- hyperparameters;
- selection stage;
- best primary value;
- strict primary-equivalence boundary.

The reconstructed temperature decision also exactly matched the stored
decision.

The artifact audit additionally confirmed:

- density result count: 72;
- temperature result count: 36;
- total results: 108;
- TEST targets accessed: no;
- numerical sanity concerns: 0.

Artifact reconstruction result:

`PHASE4D_ARTIFACT_AUDIT = PASS`

## 14. Reproducibility audit

The complete 108-fit Phase 4D validation-selection experiment was replayed
using the current frozen software environment.

Official artifact SHA-256:

`1d95d8ac7606e8a838920bea488cf4c3818eb7e072129b1e924737638eff85b2`

Replay artifact SHA-256:

`1d95d8ac7606e8a838920bea488cf4c3818eb7e072129b1e924737638eff85b2`

The replay produced:

- byte-for-byte equality: true;
- semantic JSON equality: true;
- identical density selection;
- identical density transformation;
- identical temperature selection;
- identical hyperparameters;
- identical metrics.

The temporary replay artifact was deleted after verification.

Reproducibility result:

`PHASE4D_REPRODUCIBILITY = PASS`

## 15. Interpretation

Phase 4D establishes the validation-selected classical surrogate candidates that
may proceed to Phase 4E physics-aware pre-test acceptance.

For electron density, the controlled procedure selected a log10 Extra Trees
surrogate with 500 estimators, depth limited to 16 and leaf size 1.

The absolute best density primary score was achieved by the unlimited-depth
log10 Extra Trees configuration, but the selected depth-16 log10 candidate was
within the frozen 2% primary-equivalence band and won on the first declared
secondary discriminator.

This distinction is intentional and demonstrates that the frozen selection
policy was followed rather than retrospectively choosing the numerically lowest
single metric.

For electron temperature, the unlimited-depth 500-estimator Extra Trees
configuration was the sole candidate inside the frozen 2% RMSE equivalence band
and was therefore selected directly on the primary metric.

The very small validation errors remain consistent with dense in-domain
interpolation of a smooth deterministic synthetic reduced-order simulator in a
two-dimensional qualified numerical envelope.

TRAIN and VALIDATION are independent scrambled Sobol samples from the same
qualified numerical envelope.

VALIDATION therefore represents in-domain interpolation/generalisation
assessment.

It does not constitute:

- OOD validation;
- experimental accuracy;
- industrial accuracy;
- semiconductor-process validation;
- OIPT process accuracy.

## 16. Scientific claim boundary

Phase 4D preserves the following limitations:

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

The project demonstrates transferable research-engineering capability relevant
to physics-aware AI and plasma process optimisation.

It does not claim reproduction of Oxford Instruments Plasma Technology's
industrial process-modelling system.

## 17. Phase boundary after Phase 4D

Phase 4D has selected validation-stage candidates only.

Phase 4D has not performed:

- Phase 4E physics-aware probe-grid acceptance;
- TRAIN + VALIDATION final refitting;
- Phase 4F locked TEST evaluation;
- Phase 4G production persistence;
- Phase 4G speed benchmarking.

The density and temperature candidates selected in this phase are therefore not
yet final production surrogates.

They proceed next to Phase 4E only after formal Phase 4D closure.

## 18. Phase 4D closure requirements

Before Phase 4D is formally closed:

1. the Phase 4D selection artifact must remain reviewed and reproducible;
2. the Phase 4D technical summary must be reviewed;
3. targeted Phase 4C/4D regression tests must pass;
4. the complete repository regression suite must pass;
5. staged changes must pass `git diff --cached --check`;
6. the Phase 4D source, tests, selection artifact and technical summary must be
   included in the closure commit;
7. the working tree must be clean after the closure commit.

Phase 4E must not begin until these checks pass.
