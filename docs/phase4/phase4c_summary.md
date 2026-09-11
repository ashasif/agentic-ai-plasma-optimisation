# Phase 4C - Classical Surrogate Benchmark Suite

**Status:** COMPLETE - VALIDATED AND READY FOR CLOSURE COMMIT

## 1. Scope

Phase 4C established the classical surrogate benchmark beyond the Phase 4B
dummy and ordinary-linear reference models.

The benchmark implemented and evaluated the following frozen Phase 4A
candidate families:

- polynomial response-surface regression;
- ExtraTreesRegressor;
- HistGradientBoostingRegressor.

Electron-density models were evaluated with both frozen target representations:

- identity;
- log10.

Electron-temperature models were evaluated with:

- identity only.

Phase 4C used TRAIN only for fitting and VALIDATION only for evaluation.

TEST target values were not accessed.

No final surrogate model was selected.

No final electron-density target transformation was selected.

No production surrogate was persisted.

Final controlled validation-based hyperparameter/model selection remains reserved
for Phase 4D.

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

Approved features:

- `nominal_absorbed_power_W`
- `target_pressure_mTorr`

Primary targets:

- `true_electron_density_m3`
- `true_electron_temperature_eV`

The existing Phase 4 controlled loader was reused.

TEST features remained available for structural checks, but TEST targets were
withheld by default and remained locked throughout Phase 4C.

## 3. Implemented classical-surrogate infrastructure

Phase 4C added:

`src/plasma_ai/surrogate/classical.py`

This module provides:

- explicit classical candidate specifications;
- frozen search-space guards;
- polynomial candidate construction;
- Extra Trees candidate construction;
- Histogram Gradient Boosting candidate construction;
- target-transform validation;
- TRAIN-only fitting;
- inverse transformation of predictions to physical units;
- frozen physical-scale metric calculation;
- prediction sanity metadata.

Phase 4C also added:

`src/plasma_ai/surrogate/classical_experiment.py`

This module provides:

- deterministic experiment orchestration;
- explicit TEST-lock enforcement;
- the frozen Phase 4C target/transform plan;
- a 12-fit benchmark plan;
- Phase 4B linear-baseline comparison;
- structured result-artifact generation;
- explicit metadata preventing Phase 4C results from being represented as
  Phase 4D final selection.

## 4. Phase 4C benchmark design

Four predeclared benchmark configurations were evaluated.

### 4.1 Polynomial response surfaces

Polynomial degrees:

- degree 2;
- degree 3.

Polynomial expansion includes powers and interaction terms.

The estimator after feature expansion is ordinary linear regression.

No Ridge regularisation was introduced.

### 4.2 Extra Trees reference configuration

The Phase 4C representative Extra Trees configuration was:

- `n_estimators = 200`
- `max_depth = None`
- `min_samples_leaf = 1`
- `max_features = 1.0`
- `random_state = 20260913`

This configuration lies within the frozen Phase 4A Extra Trees search space.

It is a predeclared Phase 4C benchmark point.

It was not chosen by validation-based hyperparameter optimisation.

### 4.3 Histogram Gradient Boosting reference configuration

The Phase 4C representative HistGradientBoosting configuration was:

- `learning_rate = 0.10`
- `max_iter = 200`
- `max_leaf_nodes = 31`
- `l2_regularization = 0.0`
- `random_state = 20260913`

This configuration lies within the frozen Phase 4A Histogram Gradient Boosting
search space.

It is a predeclared Phase 4C benchmark point.

It was not chosen by validation-based hyperparameter optimisation.

### 4.4 Total benchmark fits

The Phase 4C plan consisted of:

- 4 candidate configurations;
- 3 target/transform combinations;
- 12 total TRAIN-to-VALIDATION fits.

The target/transform combinations were:

1. electron density - identity;
2. electron density - log10;
3. electron temperature - identity.

The complete frozen nonlinear hyperparameter grids remain reserved for Phase 4D.

## 5. Structured benchmark artifact

Structured result artifact:

`results/phase4/classical_benchmark.json`

Artifact SHA-256 on the current Windows filesystem:

`f6cb826e83dcfecc4fc8612cbf88ae557be90455adce7f2f145d08ebc01cd2a3`

Data use:

- fitting: TRAIN only;
- evaluation: VALIDATION only;
- TEST targets accessed: no;
- final selection performed: no.

The artifact contains:

- scientific-scope metadata;
- split-use metadata;
- benchmark-design metadata;
- embedded Phase 4B reference-baseline results;
- Phase 4B linear-baseline comparisons;
- the 12 Phase 4C validation results;
- prediction-sanity metadata.

## 6. Electron-density validation results

The frozen primary density validation discriminator is P95 absolute relative
error.

### 6.1 Identity target representation

#### Polynomial degree 2

- P95 absolute relative error: 5.6870%
- median absolute relative error: 0.8953%
- R-squared: 0.999065

Relative improvement over the Phase 4B identity linear baseline on the primary
metric:

- approximately 84.52%

#### Polynomial degree 3

- P95 absolute relative error: 1.1518%
- median absolute relative error: 0.1821%
- R-squared: 0.999958

Relative improvement over the Phase 4B identity linear baseline:

- approximately 96.86%

#### Extra Trees reference

- P95 absolute relative error: 0.4550%
- median absolute relative error: 0.1040%
- R-squared: 0.999985

Relative improvement over the Phase 4B identity linear baseline:

- approximately 98.76%

#### Histogram Gradient Boosting reference

- P95 absolute relative error: 1.5710%
- median absolute relative error: 0.4249%
- R-squared: 0.999832

Relative improvement over the Phase 4B identity linear baseline:

- approximately 95.72%

### 6.2 Log10 target representation

All reported metrics below were calculated after inverse transformation back to
physical electron-density units.

#### Polynomial degree 2

- P95 absolute relative error: 6.1925%
- median absolute relative error: 2.3471%
- R-squared: 0.995383

Relative improvement over the Phase 4B log10 linear baseline:

- approximately 75.28%

#### Polynomial degree 3

- P95 absolute relative error: 1.9350%
- median absolute relative error: 0.7396%
- R-squared: 0.999539

Relative improvement over the Phase 4B log10 linear baseline:

- approximately 92.28%

#### Extra Trees reference

- P95 absolute relative error: 0.4369%
- median absolute relative error: 0.1068%
- R-squared: 0.999985

Relative improvement over the Phase 4B log10 linear baseline:

- approximately 98.26%

#### Histogram Gradient Boosting reference

- P95 absolute relative error: 1.5143%
- median absolute relative error: 0.4530%
- R-squared: 0.999802

Relative improvement over the Phase 4B log10 linear baseline:

- approximately 93.96%

## 7. Electron-temperature validation results

The frozen primary electron-temperature validation discriminator is RMSE in eV.

### 7.1 Polynomial degree 2

- RMSE: 0.012939 eV
- P95 absolute error: 0.021021 eV
- R-squared: 0.990920

Relative improvement over the Phase 4B linear baseline:

- approximately 66.70%

### 7.2 Polynomial degree 3

- RMSE: 0.004599 eV
- P95 absolute error: 0.007263 eV
- R-squared: 0.998853

Relative improvement over the Phase 4B linear baseline:

- approximately 88.16%

### 7.3 Extra Trees reference

- RMSE: 0.00001325 eV
- P95 absolute error: 0.00002533 eV
- R-squared: approximately 0.99999999

Relative improvement over the Phase 4B linear baseline:

- approximately 99.97%

### 7.4 Histogram Gradient Boosting reference

- RMSE: 0.00086275 eV
- P95 absolute error: 0.00185691 eV
- R-squared: 0.999960

Relative improvement over the Phase 4B linear baseline:

- approximately 97.78%

## 8. Comparison with Phase 4B reference baselines

Every one of the 12 Phase 4C benchmark results improved on the corresponding
Phase 4B ordinary-linear baseline using the frozen target-specific primary
validation discriminator.

Observed primary-metric relative improvements were approximately:

- density identity: 84.5% to 98.8%;
- density log10: 75.3% to 98.3%;
- electron temperature: 66.7% to 99.97%.

These comparisons demonstrate that nonlinear response structure remains after
the ordinary-linear Phase 4B baseline.

They do not constitute final model selection.

## 9. Descriptive family-level observations

The following observations are descriptive only and do not constitute the
Phase 4D selection procedure.

Within the Phase 4C polynomial family:

- degree 3 performed materially better than degree 2 for both targets;
- density identity gave a lower primary metric than density log10 for the
  degree-3 polynomial benchmark.

Within the Phase 4C Extra Trees benchmark:

- density log10 gave a slightly lower P95 absolute relative error than density
  identity;
- density identity gave a slightly lower median absolute relative error;
- electron-temperature validation error was extremely small.

Within the Phase 4C Histogram Gradient Boosting benchmark:

- density log10 gave a slightly lower P95 absolute relative error than density
  identity;
- electron-temperature error was substantially lower than both polynomial
  response surfaces.

Extra Trees produced the lowest observed primary validation metric among the
Phase 4C benchmark configurations for both targets.

This is not a final surrogate-model decision.

The full frozen parameter-space comparison remains a Phase 4D activity.

## 10. Numerical sanity checks

All 12 benchmark results passed the Phase 4C lightweight prediction-sanity
checks.

Across all validation predictions:

- all predictions were finite;
- total negative predictions: 0;
- total non-positive predictions: 0.

No polynomial numerical instability was observed.

No non-finite inverse-transform results were observed.

Full physics-aware probe-grid, monotonicity, trend and local-oscillation
assessment remains reserved for Phase 4E.

## 11. Investigation of near-perfect Extra Trees temperature interpolation

The Extra Trees electron-temperature result was sufficiently accurate to require
an explicit leakage/interpolation audit.

The audit used TRAIN and VALIDATION only.

TEST targets remained locked.

Results:

- exact TRAIN/VALIDATION feature overlap: 0 rows;
- unique TRAIN feature rows: 4,096 of 4,096;
- unique VALIDATION feature rows: 2,048 of 2,048.

TRAIN and VALIDATION both densely cover the same frozen two-dimensional
numerically qualified model envelope.

Using feature coordinates normalised by the frozen envelope, nearest-TRAIN
distance for VALIDATION points was:

- minimum: approximately 0.000118;
- 5th percentile: approximately 0.00202;
- median: approximately 0.00662;
- 95th percentile: approximately 0.01127;
- maximum: approximately 0.01557.

For the Extra Trees temperature benchmark:

- TRAIN RMSE: approximately `4.44e-15 eV`;
- VALIDATION RMSE: approximately `1.325e-05 eV`;
- VALIDATION target range: approximately `0.52695 eV`;
- validation RMSE as a fraction of target range: approximately `2.51e-05`.

The result therefore remains consistent with very dense in-domain interpolation
of a smooth deterministic synthetic reduced-order simulator.

The audit did not identify direct TRAIN/VALIDATION feature duplication.

The result must not be interpreted as experimental accuracy, industrial
accuracy, OOD generalisation, or OIPT process accuracy.

## 12. Reproducibility audit

The complete Phase 4C benchmark was replayed using the current frozen software
environment.

The replay produced:

- 12 results;
- 12 unique target/transform/candidate keys;
- zero semantic differences from the stored artifact;
- zero floating-point differences;
- exact equality of all stored primary validation metrics.

An initial raw-byte SHA comparison differed because the stored Windows artifact
uses CRLF line endings while the in-memory canonical serialization used LF line
endings.

A follow-up audit confirmed:

- text equality after normal text reading;
- semantic JSON equality;
- byte equality after CRLF-to-LF normalisation.

The discrepancy was therefore a newline-serialization difference and not model
nondeterminism or numerical variation.

## 13. Interpretation

Phase 4C demonstrates that the two approved process inputs contain sufficient
information for highly accurate surrogate interpolation of the synthetic
reduced-order simulator inside the numerically qualified model envelope.

The results also show that substantial nonlinear structure remains beyond the
ordinary-linear Phase 4B baseline.

The benchmark does not establish a final surrogate.

The benchmark does not establish a final density transformation.

The benchmark does not establish final production hyperparameters.

Those decisions remain governed by the frozen Phase 4D validation-selection
procedure.

TRAIN and VALIDATION are independent scrambled Sobol samples from the same
qualified numerical envelope.

VALIDATION therefore represents in-domain interpolation/generalisation
assessment.

It is not OOD assessment or experimental validation.

## 14. Scientific claim boundary

Phase 4C preserves the following limitations:

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

## 15. Phase 4C closure requirements

Before Phase 4C is formally closed:

1. the Phase 4C benchmark artifact must be reviewed;
2. targeted Phase 4C tests must pass;
3. the complete repository regression suite must pass;
4. staged changes must pass `git diff --cached --check`;
5. the Phase 4C source, tests, benchmark artifact and technical summary must be
   included in the closure commit;
6. the working tree must be clean after the closure commit.

Phase 4D must not begin until these checks pass.
