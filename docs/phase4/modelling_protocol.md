# Phase 4 — Surrogate Modelling & Physics-Aware Validation

## Phase 4A Modelling Protocol

**Status:** FROZEN — PHASE 4A PROTOCOL
**Project:** Agentic AI for Reduced-Order ICP Plasma Process Monitoring, Fault Diagnosis and Constrained Operating-Point Optimisation

---

## 1. Purpose

Phase 4 develops a computationally inexpensive surrogate of the frozen synthetic
reduced-order argon ICP simulator.

The surrogate approximates the in-domain mapping:

- nominal absorbed power
- target pressure

to:

- electron density
- electron temperature

The surrogate is intended to support later constrained optimisation and agentic
workflow experiments.

It is not an experimentally validated plasma model and must not be represented
as one.

---

## 2. Scientific Scope

The source simulator is a zero-dimensional reduced-order argon plasma global model.

The qualified Phase 3 operating envelope is:

| Variable | Range / value |
|---|---|
| Nominal absorbed power | 15–90 W |
| Target pressure | 10–60 mTorr |
| Argon flow | 20 sccm fixed |
| Gas temperature | 300 K |
| Effective ion-neutral cross-section | 1e-18 m² |

Important interpretation:

- absorbed power is not generator RF power;
- the model is not CFD;
- the model is not PIC;
- the model has no wafer-scale spatial resolution;
- the model contains no reactive etch or deposition chemistry;
- it does not predict etch rate, deposition rate, selectivity, uniformity,
  critical dimension, feature profile, or other semiconductor process outcomes;
- it is not calibrated against experimental OIPT data;
- it is not industrially validated;
- the qualified numerical envelope must not be described as an OIPT operating range.

Phase 4 therefore validates the surrogate against the behaviour of this specific
reduced-order simulator, not against universal ICP physics or experimental plasma
behaviour.

---

## 3. Frozen Dataset Contract

Primary dataset:

`data/synthetic/phase3/base_steady_state.csv`

Frozen SHA-256:

`dc8d3a24f84b3eca9ce930a88e969e51892d69430ce5840cc546523e9f9dd423`

Total rows:

8,192

Frozen split allocation:

| Split | Rows |
|---|---:|
| train | 4,096 |
| validation | 2,048 |
| test | 2,048 |

The existing split labels must be used exactly as frozen.

No random re-splitting is permitted.

The Phase 3 monitoring/fault dataset:

`data/synthetic/phase3/monitoring_episodes.csv`

is excluded from Phase 4 surrogate training and model selection.

Its frozen SHA-256 is:

`ee20fe9e6ffac4875911cba23247abeeeba236954349c1696cc1cb1ee59c7a3a`

Any future use of the monitoring dataset in Phase 4 would require a separate,
explicitly approved experiment and protocol amendment.

---

## 4. Frozen Feature Contract

Only the following model inputs are permitted:

1. `nominal_absorbed_power_W`
2. `target_pressure_mTorr`

No other dataset column may be automatically incorporated into the model.

In particular, truth, diagnostic, solver-status, convergence, fault, balance,
eligibility, or derived target information must not enter the feature matrix.

Feature ordering must be deterministic and preserved in persisted model metadata.

---

## 5. Frozen Target Contract

Primary targets:

1. `true_electron_density_m3`
2. `true_electron_temperature_eV`

The primary Phase 4 design will use separate target-specific surrogate models.

This is preferred because:

- the targets have very different numerical scales;
- electron density may benefit from target transformation;
- electron temperature has a comparatively narrow and strongly structured response;
- target-specific preprocessing and model selection are easier to interpret and audit.

Multi-output modelling is not part of the initial benchmark.

It may only be introduced through an explicit pre-test protocol amendment if there
is a scientifically justified reason.

---

## 6. Data-Use Policy

### 6.1 Training split

The training split may be used for:

- fitting candidate estimators;
- fitting scalers;
- fitting polynomial transformations;
- fitting target transformations;
- calculating training-only descriptive statistics.

Any learned preprocessing must be fitted using training data only during model
development.

### 6.2 Validation split

The validation split may be used for:

- algorithm comparison;
- target-transformation comparison;
- bounded hyperparameter selection;
- validation error analysis;
- model-selection decisions;
- pre-test acceptance assessment.

### 6.3 Test split

The test split is locked during model development.

Before the final configuration is frozen, test target values must not be used for:

- model comparison;
- hyperparameter selection;
- feature selection;
- transformation selection;
- architecture selection;
- stopping criteria;
- acceptance-threshold tuning.

Schema verification and verification of the frozen split membership are permitted.

The final test evaluation will occur once the complete modelling configuration has
been frozen.

After the test evaluation, the model must not be modified in response to test
performance.

Unexpected or poor test performance must be reported honestly.

---

## 7. Target Transformation Policy

### 7.1 Electron density

Two target representations will be compared using TRAIN and VALIDATION only:

1. direct modelling of `true_electron_density_m3`;
2. modelling of `log10(true_electron_density_m3)`.

The logarithmic transformation is a candidate because electron density spans a
substantial dynamic range.

It is not assumed in advance to be superior.

For transformed models:

- transformation parameters must be deterministic;
- predictions must be inverse transformed before physical-scale metrics are
  reported;
- final engineering interpretation must remain in m^-3;
- no test-set information may influence the transformation choice.

### 7.2 Electron temperature

Electron temperature will initially be modelled directly in eV.

No target log transform is planned because the source simulator produces a narrow,
structured positive temperature range.

Any later transformation would require an explicit protocol amendment before the
test set is opened.

---

## 8. Input Scaling Policy

Input scaling will be model dependent.

Scaling is expected for models whose optimisation or numerical behaviour benefits
from scaled features.

Tree-based models do not require scaling by default.

Any learned scaler must be fitted using the training split only during development
and must be included inside the persisted inference pipeline.

---

## 9. Initial Benchmark Hierarchy

The benchmark will progress from simple to more flexible models.

### Tier 0 — Reference baseline

- `DummyRegressor` using the training-target mean

Purpose:

Establish a non-informative reference and detect implementation mistakes.

### Tier 1 — Linear baseline

- ordinary linear regression

Purpose:

Measure how much of the source mapping can be represented by a first-order response.

### Tier 2 — Response-surface baseline

Polynomial regression with controlled low polynomial degree.

Initial permitted degrees:

- degree 2
- degree 3

Regularisation may be used if numerical conditioning requires it.

Purpose:

Provide an interpretable smooth response-surface baseline appropriate for the
two-dimensional deterministic input domain.

### Tier 3 — Tree ensemble

Primary ensemble family:

- `ExtraTreesRegressor`

Purpose:

Provide a flexible non-parametric classical surrogate without unnecessary deep
learning complexity.

### Tier 4 — Gradient boosting

Primary boosting family:

- `HistGradientBoostingRegressor`

Purpose:

Provide a compact nonlinear classical benchmark with controlled complexity.

### Neural-network policy

A multilayer perceptron is not part of the initial benchmark.

It may only be added before test evaluation through an explicit protocol amendment
if the classical surrogate families show a meaningful unresolved limitation on the
validation set.

The project will not add neural-network complexity solely for portfolio appearance.

---

## 10. Hyperparameter Policy

Hyperparameter exploration must be deliberately bounded.

The goal is not exhaustive optimisation.

Search spaces must:

- be documented before execution;
- remain computationally suitable for a normal laptop;
- use TRAIN for fitting;
- use VALIDATION for comparison;
- never use TEST for selection.

The initial permitted search spaces are frozen as follows.

### Linear regression

No tuned hyperparameters.

### Polynomial response surface

Polynomial degree:

- 2
- 3

Polynomial features must contain interaction terms.

The default estimator is ordinary linear regression after polynomial expansion.

If numerical-conditioning diagnostics demonstrate a genuine instability, Ridge
regression may be introduced through a documented pre-test protocol amendment.
It must not be added merely because validation accuracy is disappointing.

### Extra Trees

Permitted values:

- `n_estimators`: 200, 500
- `max_depth`: None, 8, 16
- `min_samples_leaf`: 1, 2, 4
- `max_features`: 1.0

Random seed:

`20260913`

Other estimator parameters remain at their library defaults unless a documented
pre-test protocol amendment is approved.

### Histogram Gradient Boosting

Permitted values:

- `learning_rate`: 0.05, 0.10
- `max_iter`: 200, 400
- `max_leaf_nodes`: 15, 31
- `l2_regularization`: 0.0, 0.1

Random seed:

`20260913`

Other estimator parameters remain at their library defaults unless a documented
pre-test protocol amendment is approved.

The search is a finite validation benchmark, not an attempt to discover an
optimally tuned model.

No large automated search, AutoML system, GPU requirement, or unconstrained
hyperparameter optimisation is planned.

---

## 11. Metrics

No single metric will be treated as sufficient.

### 11.1 Electron-density metrics

Reported on the physical m^-3 scale:

- MAE;
- RMSE;
- R²;
- mean absolute relative error;
- median absolute relative error;
- P95 absolute relative error;
- maximum absolute relative error.

Relative error for observation i is defined as:

`abs(predicted - true) / abs(true)`

The frozen dataset must first confirm strictly positive target values before these
relative metrics are used.

### 11.2 Electron-temperature metrics

Reported primarily in eV:

- MAE in eV;
- RMSE in eV;
- R²;
- mean absolute relative error;
- median absolute relative error;
- P95 absolute error in eV;
- maximum absolute error in eV.

Relative metrics are supplementary for electron temperature.

Absolute error in eV is the primary engineering interpretation.

---

## 12. Validation-Based Model Selection

Candidate selection must occur before test evaluation.

### Electron density

Primary validation discriminator:

- P95 absolute relative error.

Secondary discriminators, in order:

1. median absolute relative error;
2. RMSE on the physical density scale;
3. model simplicity and reproducibility.

### Electron temperature

Primary validation discriminator:

- RMSE in eV.

Secondary discriminators, in order:

1. P95 absolute error in eV;
2. MAE in eV;
3. model simplicity and reproducibility.

A numerically marginal improvement is not sufficient justification for a
substantially more complex model.

For model-complexity decisions, a change in the primary validation discriminator
smaller than 2% relative to the better score is treated as practically marginal.

When two candidates are within this 2% band on the primary discriminator, the
secondary discriminators are considered in their declared order.

If candidates remain practically indistinguishable, the simpler and more
interpretable model is preferred.

This 2% rule is a model-selection convention for this Phase 4 experiment. It is
not a physical accuracy threshold and must not be represented as one.

Physics-aware acceptance checks may reject an otherwise statistically strong model.

---

## 13. Boundary and Interior Definitions

Boundary analysis will use the known physical design limits rather than empirical
sample quantiles.

Normalized coordinates are:

`P_norm = (P - 15) / (90 - 15)`

`p_norm = (p - 10) / (60 - 10)`

A point is classified as **near-boundary** when any normalized coordinate is:

- <= 0.10, or
- >= 0.90.

A point is classified as **interior** when both normalized coordinates are:

- > 0.10, and
- < 0.90.

A stricter **central-domain** diagnostic region is defined when both normalized
coordinates lie within:

- [0.20, 0.80].

These definitions must not be changed after model results are inspected merely to
produce more favourable error summaries.

---

## 14. Operating-Space Error Analysis

Error analysis will include, at minimum:

- low, middle, and high absorbed-power regions;
- low, middle, and high pressure regions;
- interior versus near-boundary points;
- central-domain points;
- low-power/low-pressure combinations;
- low-power/high-pressure combinations;
- high-power/low-pressure combinations;
- high-power/high-pressure combinations;
- worst-error observations.

Binning rules are frozen using normalized physical-domain coordinates.

For either input coordinate x_norm in [0, 1]:

- low: 0.0 <= x_norm < 1/3;
- middle: 1/3 <= x_norm < 2/3;
- high: 2/3 <= x_norm <= 1.0.

Therefore the operating-space summaries will use physical-range thirds rather than
empirical quantiles.

Corner diagnostics are defined using the Cartesian combinations:

- low power / low pressure;
- low power / high pressure;
- high power / low pressure;
- high power / high pressure.

Here, low and high use the frozen thirds defined above.

Binning rules must not be changed after residuals or validation-model comparisons
are inspected.

Future heatmaps or dense-grid visualisations must be described as interpolation
diagnostics within the numerically qualified source-model domain.

They must not be described as OOD validation.

---

## 15. Physics-Aware Validation

Aggregate statistical accuracy is not sufficient.

The selected surrogate must also reproduce important structural behaviour observed
in the frozen reduced-order simulator.

Within the qualified model domain, diagnostics will examine whether:

### At fixed pressure

Increasing absorbed power reproduces the source simulator's strong increase in
electron density.

### At fixed absorbed power

Increasing pressure reproduces the source simulator's:

- increase in electron density;
- decrease in electron temperature.

These statements describe the behaviour of this specific reduced-order simulator.

They must not be represented as universal experimental ICP laws.

Additional checks will include:

- predicted electron density must remain positive;
- predicted electron temperature must remain positive;
- no obviously spurious local oscillation should appear in a smooth source-model
  response;
- monotonic trend violations will be counted and localised.

The deterministic physics-validation probe design is frozen as a regular
41 x 41 Cartesian grid spanning the complete qualified source-model domain:

- absorbed power: 41 equally spaced values from 15 W through 90 W, inclusive;
- target pressure: 41 equally spaced values from 10 mTorr through 60 mTorr,
  inclusive.

This produces 1,681 deterministic in-domain probe points.

The source simulator will be evaluated on this grid independently of surrogate
training. These simulator evaluations are diagnostic reference calculations and
must not be appended to TRAIN, VALIDATION, or TEST.

Trend checks will operate along the grid axes:

- for every fixed-pressure grid line, inspect electron density as absorbed power
  increases;
- for every fixed-power grid line, inspect electron density as pressure increases;
- for every fixed-power grid line, inspect electron temperature as pressure
  increases.

The source simulator's own grid behaviour must be recorded first. A surrogate will
not be penalised for failing a monotonic relationship that the source simulator
itself does not satisfy on the corresponding diagnostic line.

Surrogate monotonic violations will therefore always be interpreted relative to
the measured source-model behaviour on the same frozen grid.

Such probes are interpolation diagnostics, not experimental validation and not
OOD validation.

---

## 16. Pre-Test Model Lock

Before the test set is evaluated:

1. the final algorithm must be selected;
2. all hyperparameters must be frozen;
3. target-transformation choice must be frozen;
4. preprocessing must be frozen;
5. feature ordering must be frozen;
6. physics-aware validation must be completed;
7. boundary/interior validation behaviour must be reviewed;
8. model metadata schema must be defined;
9. random seeds must be frozen.

Only after these steps pass may the locked final model configuration proceed to
test evaluation.

---

## 17. Final Refit Policy

After model and transformation selection using TRAIN and VALIDATION:

- the selected configuration may be refitted using TRAIN + VALIDATION;
- no configuration changes are allowed during this refit;
- the resulting fitted model becomes the locked final surrogate candidate;
- TEST is then used once for final generalisation evaluation.

The combined refit dataset therefore contains:

6,144 rows.

TEST remains:

2,048 rows.

---

## 18. Locked Test Evaluation

The final locked test evaluation will report the complete predeclared metric set.

It will also include:

- boundary/interior error summaries;
- operating-space error summaries;
- worst-case test points;
- physical positivity checks;
- structural trend diagnostics where appropriate.

No test result may trigger model retuning within Phase 4.

If a major defect is discovered after opening TEST, it must be documented
explicitly and handled as a new experimental iteration rather than silently
reusing the same test set as validation data.

---

## 19. Speed Benchmark Protocol

The final surrogate will be benchmarked against the original reduced-order solver.

The benchmark specification must be frozen before timing.

It will record:

- CPU and platform information;
- Python version;
- relevant package versions;
- number of evaluation points;
- scalar versus batch prediction where applicable;
- warm-up policy;
- repetition count;
- timing function;
- median or robust summary runtime;
- exactly which simulator and surrogate operations are included.

Speed-up will be calculated only from comparable workloads.

Runtime results will be described as hardware and implementation dependent.

No industrial throughput claim will be made.

---

## 20. Model Persistence and Provenance

Final persisted surrogate artifacts must include enough metadata to reconstruct
their meaning.

At minimum:

- model type;
- target name;
- feature names and ordering;
- feature units;
- target unit;
- target transformation;
- hyperparameters;
- training-data identity;
- frozen Phase 3 dataset SHA-256;
- rows used for final fitting;
- source Git commit;
- Python version;
- key package versions;
- training timestamp;
- random seed where applicable;
- qualified input bounds;
- scientific limitations.

Persistence must include any preprocessing required for inference.

---

## 21. Reproducibility

Phase 4 must provide:

- deterministic data loading;
- contract validation;
- controlled random seeds;
- reproducible model training;
- reproducible evaluation;
- persisted configuration;
- automated tests;
- documented provenance.

Generated results must not be committed until validated.

---

## 22. Phase 4 Subphase Structure

### Phase 4A
Surrogate experiment specification and frozen modelling protocol.

### Phase 4B
Dataset loader, integrity checks, train-only transformations, and reference
baselines.

### Phase 4C
Classical surrogate benchmark suite using TRAIN and VALIDATION only.

### Phase 4D
Controlled validation-based tuning and final model selection.

### Phase 4E
Pre-test physics-aware validation, error-space analysis, acceptance checks,
configuration lock, and final TRAIN + VALIDATION refit.

### Phase 4F
One-time locked TEST evaluation.

### Phase 4G
Inference-speed benchmark, persistence, provenance, and reproducible inference.

### Phase 4H
Documentation, automated validation, reproducibility audit, scientific limitations,
formal Phase 4 closure, and final Git checkpoint.

No Phase 5, Phase 6, or Phase 7 work is permitted during Phase 4.

---

## 23. Scientific Claim Boundary

All Phase 4 reporting must preserve the following language:

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

The purpose of Phase 4 is to demonstrate transferable research-engineering
capability in physics-aware surrogate modelling and validation.

It does not claim reproduction of Oxford Instruments Plasma Technology's industrial
process-modelling system.
