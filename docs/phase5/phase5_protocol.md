# Phase 5 Protocol

## Status

**Phase:** 5

**Title:** Process Monitoring, Anomaly Detection and Fault Diagnosis

**Protocol state:** FROZEN

**Implementation status:** NOT STARTED

**TEST status:** NOT USED FOR MODEL DEVELOPMENT OR PERFORMANCE EVALUATION

This protocol was reviewed and frozen before Phase 5 implementation,
model fitting, validation-result inspection, model selection or predictive
TEST evaluation.

Phase 5 implementation remains prohibited until the dedicated protocol-freeze
Git commit has been created and the post-commit working tree is clean.

---

## 1. Purpose

Phase 5 converts the frozen Phase 3 synthetic monitoring environment into a
deterministic, leakage-safe and auditable process-monitoring and fault-diagnosis
workflow.

The phase has three scientific functions:

1. establish healthy-reference anomaly-monitoring baselines;
2. detect whether a synthetic fault effect is active;
3. diagnose the active synthetic fault family when a fault is active.

Phase 5 does not perform operating-point optimisation and does not implement an
agentic decision-making system.

---

## 2. Authoritative upstream evidence

Phase 5 is based on the following frozen Phase 3 monitoring artifacts:

- `data/synthetic/phase3/monitoring_episodes.csv`
- `configs/phase3/monitoring_dataset.json`
- `configs/phase3/monitoring_feature_manifest.json`
- `artifacts/phase3/monitoring_manifest.json`
- `artifacts/phase3/monitoring_summary.json`

Frozen monitoring dataset SHA-256:

`ee20fe9e6ffac4875911cba23247abeeeba236954349c1696cc1cb1ee59c7a3a`

Frozen monitoring feature-manifest SHA-256:

`ece31a97b317bda2da3a366dec1fb69d8756e05f3da64b10268d033dabbe948b`

Dataset structure:

- 4,096 rows;
- 64 episodes;
- 64 ordered quasi-steady steps per episode;
- TRAIN: 32 episodes / 2,048 rows;
- VALIDATION: 16 episodes / 1,024 rows;
- TEST: 16 episodes / 1,024 rows;
- no episode is shared across splits.

The ordered step index is not calibrated physical time.

---

## 3. Scientific scope

The monitoring environment is a synthetic quasi-steady reduced-order argon
plasma simulation.

The synthetic fault families are:

- `power_coupling`;
- `flow_delivery`;
- `pumping_effectiveness`;
- `pressure_sensor_bias`.

Only single-fault scenarios are represented.

Compound faults are outside Phase 5 scope.

Fault profiles are:

- step;
- smooth drift.

Fault severity levels are:

- mild;
- moderate;
- severe.

Fault onset occurs between ordered steps 16 and 31.

The dataset also contains normal episodes and pre-effect rows from fault
episodes.

---

## 4. Scientific claim boundary

Phase 5 results must retain all of the following limitations:

- synthetic data only;
- reduced-order argon plasma model;
- no experimental plasma validation;
- no industrial validation;
- no OIPT operating-range claim;
- absorbed power is not generator RF power;
- no reactive etch or deposition prediction;
- no wafer-scale spatial plasma modelling;
- ordered steps are not calibrated physical time;
- synthetic process variability, noise and fault magnitudes are scenario
  assumptions;
- successful detection or diagnosis demonstrates performance on this frozen
  synthetic environment only.

Phase 5 must not describe fault-detection delay in seconds or another physical
time unit.

---

## 5. Frozen predictive input contract

The only approved raw predictive inputs are:

1. `nominal_absorbed_power_W`
2. `target_pressure_mTorr`
3. `nominal_flow_sccm`
4. `measured_absorbed_power_W`
5. `measured_flow_sccm`
6. `measured_pressure_mTorr`

Numeric-column auto-selection is forbidden.

The Phase 3 feature manifest must be loaded and validated before model
development.

---

## 6. Frozen deterministic derived features

Phase 5 may additionally derive exactly three row-local residual features:

`absorbed_power_relative_residual`

defined as:

`measured_absorbed_power_W / nominal_absorbed_power_W - 1`

`flow_relative_residual`

defined as:

`measured_flow_sccm / nominal_flow_sccm - 1`

`pressure_relative_residual`

defined as:

`measured_pressure_mTorr / target_pressure_mTorr - 1`

These derived features use only approved observable/nominal inputs.

No future rows, centred windows, backward-looking target labels, protected
ground truth or simulator state may be used to derive a predictive feature.

The frozen Phase 5 model feature order is therefore:

1. `nominal_absorbed_power_W`
2. `target_pressure_mTorr`
3. `nominal_flow_sccm`
4. `measured_absorbed_power_W`
5. `measured_flow_sccm`
6. `measured_pressure_mTorr`
7. `absorbed_power_relative_residual`
8. `flow_relative_residual`
9. `pressure_relative_residual`

---

## 7. Explicitly forbidden predictive inputs

The following are not predictive features:

- `episode_id`;
- `split`;
- `step_index`;
- sequence identifiers;
- fault planning metadata;
- fault labels;
- fault magnitude or progression fields;
- simulator truth;
- plasma-state truth;
- latent process factors;
- measurement-noise realization fields;
- solver/qualification status fields;
- any TEST-derived statistic;
- any feature using future sequence rows.

Every field listed in the Phase 3 `protected_ground_truth` contract remains
protected.

`step_index` may be used only for ordering and post-prediction monitoring-delay
analysis.

`episode_id` and `split` may be used only for grouping and governance.

---

## 8. Prediction targets

### 8.1 Binary fault detection

Primary binary target:

`fault_effect_active`

The binary task is evaluated on all eligible rows.

### 8.2 Active-fault diagnosis

Primary diagnosis target:

`active_fault_family`

Primary fault-family diagnosis is evaluated only where
`fault_effect_active == True`.

The active diagnosis classes are:

- `flow_delivery`;
- `power_coupling`;
- `pressure_sensor_bias`;
- `pumping_effectiveness`.

### 8.3 End-to-end classification

An end-to-end five-state result may also be reported:

- `none`;
- `flow_delivery`;
- `power_coupling`;
- `pressure_sensor_bias`;
- `pumping_effectiveness`.

This is a secondary system-level metric and must not replace the primary binary
detection and active-fault diagnosis metrics.

---

## 9. Structural identifiability caution

`pumping_effectiveness` and `pressure_sensor_bias` may produce partially
confounded measured-pressure signatures because Phase 5 intentionally does not
fabricate an independent electron-density or electron-temperature diagnostic.

Per-class confusion must therefore be reported.

Poor separability must be reported as a scientific limitation rather than
hidden by aggregate accuracy.

Acceptance criteria may not be weakened after validation results are observed.

---

## 10. Split discipline

TRAIN, VALIDATION and TEST assignments are frozen at episode level.

TRAIN:

- 32 episodes;
- 2,048 rows.

VALIDATION:

- 16 episodes;
- 1,024 rows.

TEST:

- 16 episodes;
- 1,024 rows.

No row may move between splits.

No episode may appear in more than one split.

All preprocessing parameters used during model fitting must be learned from
permitted development data only.

---

## 11. TRAIN discipline

TRAIN may be used for:

- fitting candidate models;
- fitting scalers;
- estimating healthy-reference statistics;
- debugging deterministic implementation;
- unit/integration testing using non-TEST data;
- controlled model-development diagnostics.

TRAIN may not be expanded using VALIDATION or TEST during candidate
development.

No TEST feature matrix may be passed into model-development code.

---

## 12. VALIDATION discipline

VALIDATION may be used for:

- comparing the frozen candidate configurations;
- selecting the primary binary detector;
- selecting the primary active-fault diagnoser;
- evaluating the frozen healthy-reference anomaly baselines;
- determining whether Phase 5 acceptance criteria are satisfied.

Candidate configurations and acceptance criteria must be frozen before any
Phase 5 validation model result is inspected.

Once Phase 5E selection is complete, the winning configurations are frozen.

---

## 13. TEST discipline

The Phase 5 TEST split is reserved for one final locked performance evaluation
in Phase 5F.

Before protocol freeze, Gate 2B performed a read-only structural audit that
reported aggregate TEST split metadata and aggregate target counts.

That audit did not:

- train a model on TEST;
- generate TEST predictions;
- calculate TEST predictive performance;
- inspect model residuals on TEST;
- tune a threshold from TEST;
- select a model from TEST evidence.

After this protocol is frozen, there must be no further TEST feature or target
access for model development, candidate comparison, threshold design or
acceptance-rule changes.

If Phase 5 validation acceptance fails, Phase 5F must not run.

Any redevelopment after a failed validation gate must remain TRAIN/VALIDATION
only and must be separately governed before TEST access.

After the Phase 5F evaluator runs successfully, the Phase 5 TEST split is
considered consumed for this development cycle.

The locked TEST evaluator must not be rerun merely to improve reported
performance.

---

## 14. Phase 5C healthy-reference anomaly monitoring

Phase 5C establishes two frozen reference anomaly-detection approaches.

### 14.1 Robust residual score

The healthy reference is fitted from TRAIN rows where
`fault_effect_active == False`.

The score uses the three frozen relative residual features.

For each residual, TRAIN-healthy median and median absolute deviation are
calculated.

Robust scale is:

`1.4826 * MAD`

If a MAD is numerically zero, the TRAIN-healthy standard deviation is used as
the deterministic fallback.

The row anomaly score is the maximum absolute robust z-score across the three
residuals.

The anomaly threshold is the 0.99 quantile of TRAIN-healthy scores.

### 14.2 Isolation Forest reference

Frozen configuration:

- `n_estimators = 500`;
- `contamination = 0.01`;
- `random_state = 20260924`;
- `n_jobs = 1`.

The Isolation Forest is fitted on TRAIN healthy rows only using the nine frozen
Phase 5 model features.

### 14.3 Anomaly baseline role

These anomaly approaches provide healthy-reference monitoring evidence.

They are not allowed to override the supervised Phase 5 detection acceptance
gate.

Validation reporting must include:

- AUROC where defined;
- average precision;
- sensitivity/recall;
- specificity;
- false-positive rate;
- confusion counts.

---

## 15. Phase 5D binary detector candidates

Exactly three initial supervised detector configurations are permitted.

### Candidate D1 - logistic regression

Pipeline:

- `StandardScaler`;
- `LogisticRegression`.

Frozen classifier parameters:

- `C = 1.0`;
- `solver = "lbfgs"`;
- `max_iter = 2000`;
- `class_weight = "balanced"`.

### Candidate D2 - histogram gradient boosting

Frozen classifier parameters:

- `learning_rate = 0.05`;
- `max_iter = 300`;
- `max_leaf_nodes = 15`;
- `l2_regularization = 0.1`;
- `early_stopping = False`;
- `random_state = 20260924`.

### Candidate D3 - extra trees

Frozen classifier parameters:

- `n_estimators = 500`;
- `max_depth = None`;
- `min_samples_leaf = 1`;
- `max_features = 1.0`;
- `class_weight = "balanced"`;
- `n_jobs = 1`;
- `random_state = 20260924`.

No unplanned hyperparameter search is permitted.

The binary classification threshold for probabilistic supervised candidates is
fixed at 0.5.

No threshold optimisation from TEST is permitted.

---

## 16. Phase 5D active-fault diagnosis candidates

Exactly three initial active-fault diagnosis configurations are permitted.

Candidate G1:

- StandardScaler;
- LogisticRegression;
- `C = 1.0`;
- `solver = "lbfgs"`;
- `max_iter = 2000`;
- `class_weight = "balanced"`.

Candidate G2:

- HistGradientBoostingClassifier;
- `learning_rate = 0.05`;
- `max_iter = 300`;
- `max_leaf_nodes = 15`;
- `l2_regularization = 0.1`;
- `early_stopping = False`;
- `random_state = 20260924`.

Candidate G3:

- ExtraTreesClassifier;
- `n_estimators = 500`;
- `max_depth = None`;
- `min_samples_leaf = 1`;
- `max_features = 1.0`;
- `class_weight = "balanced"`;
- `n_jobs = 1`;
- `random_state = 20260924`.

These models are fitted only to TRAIN rows where
`fault_effect_active == True`.

No unplanned hyperparameter search is permitted.

---

## 17. Required binary-detection metrics

At minimum, report:

- balanced accuracy;
- macro F1;
- fault-active precision;
- fault-active recall;
- specificity;
- false-positive rate;
- AUROC;
- average precision;
- confusion matrix;
- per-active-fault-family detection recall.

Sequence-aware reporting must also include:

- first active step;
- first alarm at or after first active step;
- detection delay in ordered steps;
- missed-fault episode count;
- false-alarm episode count before active effect.

Detection delay is an ordered-step metric, not a physical-time metric.

---

## 18. Required active-fault diagnosis metrics

At minimum, report on ground-truth active rows:

- macro F1;
- balanced accuracy;
- per-class precision;
- per-class recall;
- per-class F1;
- confusion matrix;
- support by class.

The majority-class diagnostic baseline must also be reported.

---

## 19. Required end-to-end metrics

The frozen detector and diagnoser must also be composed into an end-to-end
five-state diagnostic pipeline.

At minimum, report:

- macro F1;
- balanced accuracy;
- per-class precision/recall/F1;
- confusion matrix.

These are secondary metrics.

---

## 20. Validation acceptance criteria

### 20.1 Binary detector

A binary detector is acceptable only if all of the following VALIDATION
conditions are met:

- balanced accuracy >= 0.75;
- macro F1 >= 0.75;
- fault-active recall >= 0.75;
- specificity >= 0.75;
- active-row detection recall for every represented fault family >= 0.60.

### 20.2 Active-fault diagnoser

An active-fault diagnoser is acceptable only if all of the following VALIDATION
conditions are met:

- macro F1 >= 0.50;
- balanced accuracy >= 0.50;
- recall for every active fault family >= 0.25;
- macro F1 improves over the majority-class diagnostic baseline by at least
  0.10.

### 20.3 Controlled stop

If no frozen detector candidate passes the binary criteria, Phase 5 stops
before TEST.

If no frozen diagnoser candidate passes the diagnosis criteria, Phase 5 stops
before TEST.

The failure must be documented.

Acceptance thresholds must not be relaxed after results are observed merely to
force progression.

Any redevelopment must be separately governed and remain TRAIN/VALIDATION only.

---

## 21. Validation selection rules

### Binary detector selection

Among candidates satisfying all binary acceptance criteria:

1. choose highest balanced accuracy;
2. if tied, choose highest macro F1;
3. if tied, choose lowest false-positive rate;
4. if still tied, use candidate priority D1, then D2, then D3.

### Active-fault diagnoser selection

Among candidates satisfying all diagnosis acceptance criteria:

1. choose highest macro F1;
2. if tied, choose highest balanced accuracy;
3. if tied, choose highest minimum per-class recall;
4. if still tied, use candidate priority G1, then G2, then G3.

Selection logic must be implemented deterministically.

---

## 22. Final refit before locked TEST

Only after Phase 5E acceptance and selection may the selected configurations be
refitted.

Final-fit development data:

- TRAIN plus VALIDATION;
- 48 episodes;
- 3,072 total rows for the detector;
- only active TRAIN plus VALIDATION rows for the diagnoser.

TEST rows used for fitting:

`0`

All selected model configurations, feature order and preprocessing rules remain
unchanged during final refit.

No new model-family selection occurs after final refit.

---

## 23. Phase 5F locked TEST evaluation

Phase 5F performs one final evaluation using the frozen final-fit models.

The evaluator must:

- verify the protocol/config state;
- verify the monitoring dataset hash;
- verify feature order;
- verify persisted/final model identity;
- load TEST only inside the locked evaluator;
- produce all required binary metrics;
- produce all required active-diagnosis metrics;
- produce all required end-to-end metrics;
- produce ordered-step monitoring-delay metrics;
- write one frozen machine-readable result artifact.

The proposed result path is:

`results/phase5/locked_test_evaluation.json`

Once frozen, its SHA-256 must be recorded in Phase 5 closure documentation.

---

## 24. Persistence and reproducibility

Phase 5G must persist the accepted final diagnostic components.

Proposed artifacts:

- `artifacts/phase5/fault_detector.pkl`
- `artifacts/phase5/fault_diagnoser.pkl`
- `artifacts/phase5/diagnostics_manifest.json`

The manifest must include:

- exact feature order;
- derived-feature definitions;
- model classes;
- model parameters;
- preprocessing contract;
- training split provenance;
- final-fit row/episode counts;
- software versions;
- random states;
- source dataset hashes;
- model artifact hashes;
- scientific limitations.

Persistence equivalence must demonstrate that predictions from the in-memory
final models and reloaded persisted models are identical on a frozen
non-TEST reproducibility fixture.

Phase 5G persistence verification must not rerun the locked TEST evaluation.

---

## 25. Relationship to the Phase 4 surrogate

The persisted Phase 4 surrogate remains frozen.

Phase 5 does not use Phase 4 TEST evidence as development feedback.

The Phase 4 models are not retrained, modified or replaced in Phase 5.

Phase 5 does not use predicted electron density or electron temperature as
diagnostic model inputs because the monitoring environment contains no
corresponding measured plasma diagnostic with which to form an observable
residual.

The Phase 4 surrogate manifest may be read for provenance and domain context.

Phase 4 surrogate outputs may not be introduced as new Phase 5 predictive
features without a separately reviewed and frozen protocol amendment.

---

## 26. Phase 6 boundary

Phase 6 is defined as:

**Constrained Operating-Point Optimisation**

Its primary computational dependency is the frozen Phase 4 surrogate.

Phase 6 may design and evaluate optimisation objectives, feasibility rules,
constraints, search strategies and operating-point recommendations inside the
qualified numerical envelope.

Phase 6 must not:

- reopen Phase 5 model selection;
- retune Phase 5 diagnostics from Phase 5 TEST;
- claim autonomous agentic operation;
- expand the Phase 4 numerical domain without a new validation protocol.

Phase 5 fault-diagnosis outputs are not required for standalone Phase 6
optimiser development.

Integration between diagnosis and optimisation belongs to Phase 7.

---

## 27. Phase 7 boundary

Phase 7 is defined as:

**Agentic Workflow Integration, Decision Orchestration and Final Demonstration**

Phase 7 may integrate the frozen capabilities from:

- Phase 5 monitoring/detection/diagnosis;
- Phase 6 constrained optimisation;
- existing frozen physics/surrogate components.

The Phase 7 agent may:

- inspect monitoring evidence;
- call diagnostic tools;
- call constrained optimisation tools;
- enforce frozen boundaries and constraints;
- generate auditable recommendations;
- explain tool calls and evidence;
- demonstrate controlled end-to-end synthetic scenarios.

Phase 7 must not silently retrain Phase 4, Phase 5 or Phase 6 components.

Agentic orchestration is therefore deliberately deferred until after both the
diagnostic and optimisation capabilities are independently frozen.

---

## 28. Phase 5 sub-phases

### Phase 5A - Protocol and governance freeze

Freeze:

- scope;
- dataset identity;
- feature contract;
- derived features;
- split discipline;
- candidate configurations;
- metrics;
- acceptance criteria;
- TEST discipline;
- Phase 5/6/7 boundaries.

### Phase 5B - Leakage-safe monitoring pipeline

Implement and test:

- dataset loader;
- hash verification;
- feature-manifest enforcement;
- split guards;
- protected-ground-truth guards;
- deterministic residual features;
- episode/order integrity checks.

No model selection occurs in 5B.

### Phase 5C - Healthy-reference anomaly monitoring

Implement and evaluate on TRAIN/VALIDATION:

- robust residual anomaly score;
- Isolation Forest reference.

No TEST access occurs.

### Phase 5D - Supervised detection and diagnosis benchmark

Fit the frozen detector and diagnoser candidates on TRAIN only.

Evaluate candidate evidence on VALIDATION according to the frozen protocol.

### Phase 5E - Validation selection and final refit

Apply frozen acceptance and deterministic selection rules.

If acceptance fails, stop before TEST.

If acceptance passes:

- freeze selected configurations;
- refit selected configurations on TRAIN plus VALIDATION;
- freeze final-fit provenance.

### Phase 5F - Locked TEST evaluation

Evaluate the final frozen diagnostic system exactly once on TEST.

Freeze result artifact and hash.

### Phase 5G - Persistence and reproducible inference

Persist diagnostic models and manifest.

Verify reload equivalence without rerunning TEST.

### Phase 5H - Final Phase 5 audit and closure

Audit:

- protocol compliance;
- frozen artifact identity;
- TEST discipline;
- reproducibility;
- scientific claims;
- repository regression;
- documentation;
- Git state.

Only after 5H passes may Phase 6 be formally unlocked.

---

## 29. Required Phase 5 evidence

Expected machine-readable evidence includes:

- `results/phase5/anomaly_baselines.json`;
- `results/phase5/supervised_benchmark.json`;
- `results/phase5/validation_selection.json`;
- final-refit provenance;
- `results/phase5/locked_test_evaluation.json`;
- persistence-equivalence evidence;
- final diagnostics manifest.

Exact schemas are to be implemented under this frozen protocol without
weakening its scientific or governance constraints.

---

## 30. Reproducibility rules

All stochastic estimators must use explicit random states.

Initial Phase 5 random state:

`20260924`

Parallel estimator execution must use deterministic settings where practical.

Initial tree-model policy:

`n_jobs = 1`

Every persisted artifact must record sufficient provenance to reconstruct its
development context.

---

## 31. Protocol-change policy

Before Phase 5 validation results are inspected, this protocol may be changed
only through explicit review.

After protocol freeze, any scientifically material change requires:

1. written justification;
2. a protocol amendment;
3. verification that TEST has not been used as feedback;
4. a dedicated Git commit before running the changed experiment.

After Phase 5F TEST consumption, no change may use the consumed TEST as tuning
feedback.

A new development cycle would require separately governed evidence.

---

## 32. Phase 5 entry criteria

Phase 5 implementation may start only when:

- Phase 4 is formally closed;
- Phase 5 is formally unlocked;
- Phase 3 monitoring hashes are verified;
- feature/split contracts are verified;
- this protocol is marked FROZEN;
- its machine-readable configuration is valid;
- protocol review checks pass;
- the protocol freeze is committed;
- the working tree is clean.

---

## 33. Phase 5 exit criteria

Phase 5 may close only when:

- every completed sub-phase is documented;
- no protected-ground-truth leakage occurred;
- no episode split leakage occurred;
- validation acceptance was applied exactly as frozen;
- TEST discipline is documented;
- if TEST was consumed, its result is frozen and hashed;
- persisted model artifacts are reproducible;
- scientific limitations are preserved;
- the complete repository regression passes;
- closure documentation is complete;
- the working tree is clean;
- Phase 6 is explicitly unlocked by the final closure checkpoint.

---

## 34. Current state

At protocol freeze:

- Phase 4 is formally closed;
- Phase 5 is unlocked;
- Phase 5 implementation has not begun;
- no Phase 5 predictive model has been trained;
- no Phase 5 validation model result has been inspected;
- no Phase 5 TEST predictive performance has been calculated;
- Gate 2B inspected aggregate TEST metadata/target counts only;
- Phase 5 TEST remains prohibited from development feedback.

The acceptance thresholds in this protocol are project governance criteria for
this synthetic research-engineering workflow. They are not experimentally
established industrial acceptance requirements.

Implementation remains blocked until this frozen protocol is committed in Git
and the repository returns to a clean state.
