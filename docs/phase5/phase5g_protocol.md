# Phase 5G — Persistence & Reproducible Inference Protocol

## Status

This protocol is **FROZEN before Phase 5G implementation**.

Phase 5F is formally closed. The Phase 5 TEST holdout has already been consumed exactly once and is permanently unavailable for further predictive evaluation or development feedback.

Phase 5G is therefore an engineering/reproducibility phase, not a modelling phase.

## Objective

Phase 5G will provide a trusted runtime interface around the already-frozen Phase 5 fault detector and fault diagnoser.

It will:

- verify the exact frozen model artifact bytes;
- record those artifacts in a runtime manifest;
- verify hashes before pickle deserialization;
- validate the exact frozen wrapper and estimator contracts;
- construct the frozen nine-feature model representation from six raw monitoring inputs;
- expose deterministic fault-detection and ambiguity-aware diagnosis inference;
- demonstrate exact persistence/runtime prediction equivalence on an independent deterministic synthetic probe set.

Phase 5G will not:

- retrain either model;
- reconstruct either model;
- reserialize either model;
- copy or replace the frozen model bytes;
- tune the detector threshold;
- change diagnosis classes;
- reopen the Phase 5 TEST dataset;
- calculate new TEST predictions;
- calculate new TEST performance;
- use Phase 5F evidence as model-development feedback.

## Frozen model artifacts

### Detector

Artifact:

`artifacts/phase5/fault_detector.pkl`

SHA-256:

`4a00e785d7e7f0a8429020399967cdd82c18da275d58e5b83bc11707eeac8a6a`

Candidate:

`ERD4_extra_trees_threshold_040`

Threshold:

`0.40`

Fit rows:

`3072`

Wrapper:

`plasma_ai.monitoring.supervised.FittedDetectorCandidate`

Estimator:

`ExtraTreesClassifier`

### Diagnoser

Artifact:

`artifacts/phase5/fault_diagnoser.pkl`

SHA-256:

`2e9dea016137594c707ecd3c32116691cc90d0127133f11c70372c93e6fcca84`

Candidate:

`ERG3_extra_trees`

Fit rows:

`1451`

Wrapper:

`plasma_ai.monitoring.redevelopment.FittedRedevelopmentDiagnoserCandidate`

Observable diagnosis classes:

1. `flow_delivery`
2. `power_coupling`
3. `pressure_path_anomaly`

`pressure_path_anomaly` remains an ambiguity-aware observable class. It must not be presented as reliable identification of a unique underlying physical mechanism.

## Artifact policy

The Phase 5 detector and diagnoser are already persisted and frozen.

Unlike Phase 4G, Phase 5G must **not reconstruct or serialize new model artifacts**.

The existing pickle bytes are authoritative.

Phase 5G will create only the new runtime manifest:

`artifacts/phase5/monitoring_manifest.json`

The manifest will reference and hash the existing model files.

Existing production artifacts must not be overwritten.

## Runtime input contract

The public inference interface accepts the six frozen raw monitoring features in this order:

1. `nominal_absorbed_power_W`
2. `target_pressure_mTorr`
3. `nominal_flow_sccm`
4. `measured_absorbed_power_W`
5. `measured_flow_sccm`
6. `measured_pressure_mTorr`

A single vector of shape `(6,)` or a batch of shape `(n_rows, 6)` is permitted.

The input must:

- be convertible to `float64`;
- contain only finite values;
- contain at least one row;
- have positive nominal absorbed power;
- have positive target pressure;
- have positive nominal flow.

No targets, fault labels, protected truth variables, episode identifiers or TEST information are runtime inputs.

## Frozen feature construction

Phase 5G constructs these three row-local residuals:

`absorbed_power_relative_residual = measured_absorbed_power_W / nominal_absorbed_power_W - 1`

`flow_relative_residual = measured_flow_sccm / nominal_flow_sccm - 1`

`pressure_relative_residual = measured_pressure_mTorr / target_pressure_mTorr - 1`

The resulting frozen nine-feature order is:

1. `nominal_absorbed_power_W`
2. `target_pressure_mTorr`
3. `nominal_flow_sccm`
4. `measured_absorbed_power_W`
5. `measured_flow_sccm`
6. `measured_pressure_mTorr`
7. `absorbed_power_relative_residual`
8. `flow_relative_residual`
9. `pressure_relative_residual`

No history, future rows, centred windows, target-derived features, simulator truth or protected design variables may be introduced.

## Runtime prediction semantics

The detector first produces active-fault probability.

The frozen binary decision rule is:

`active_probability >= 0.40`

The diagnoser is executed only on detector-positive rows (rows classified active by the detector).

The final runtime state is one of:

- `none`
- `flow_delivery`
- `power_coupling`
- `pressure_path_anomaly`

Inactive detector rows map directly to `none`.

The runtime prediction result will expose:

- `active_probability`
- `fault_active`
- `diagnostic_state`

Output row order must exactly preserve input row order.

## Trusted loading

Before any pickle deserialization:

1. load and validate the frozen Phase 5G protocol;
2. load and validate the runtime manifest;
3. verify each artifact SHA-256;
4. only then deserialize trusted project artifacts.

The loader must reject:

- missing model files;
- hash mismatch;
- unexpected wrapper classes;
- unexpected candidate IDs;
- unexpected fit-row counts;
- detector threshold drift;
- unexpected estimator type;
- estimator parameter drift;
- unexpected diagnostic classes;
- incompatible scientific-runtime versions.

The loader is for exact trusted project artifacts only. It must not be presented as a safe loader for arbitrary pickle files.

## Runtime compatibility

The frozen scientific runtime is:

- Python `3.12.5`
- NumPy `2.5.3`
- SciPy `1.18.1`
- scikit-learn `1.9.1`

Exact versions are required for Phase 5G reproducibility qualification.

Python implementation and platform are recorded as provenance.

An exact operating-system/platform match is not a frozen loading requirement.

## Reproducibility probe

Persistence equivalence uses a deterministic synthetic probe only.

It is not:

- TRAIN data;
- VALIDATION data;
- TEST data;
- a new performance benchmark;
- experimental validation;
- industrial validation;
- OOD validation.

The probe uses:

Absorbed-power values:

`25, 50, 80 W`

Target-pressure values:

`20, 32.5, 45 mTorr`

Nominal flow:

`20 sccm`

Five residual scenarios:

1. nominal;
2. absorbed power `-10%`;
3. flow `-7%`;
4. pressure `+10%`;
5. pressure `-10%`.

The Cartesian design therefore contains exactly:

`3 × 3 × 5 = 45 rows`

No labels or truth targets are assigned.

## Persistence-equivalence acceptance

Reference inference is generated by directly loading the exact frozen model bytes after hash verification.

Runtime inference is generated through the Phase 5G manifest-verified loader.

Acceptance requires exact prediction equality for the frozen inference outputs and exact equality for the constructed feature matrix:

- constructed model-feature matrix;
- detector active probabilities;
- detector binary decisions;
- diagnoser predictions;
- final four-state end-to-end predictions.

All probabilities must be finite.

No tolerance-based substitution is permitted for the exact-equality requirements.

The evidence artifact will be:

`results/phase5/persistence_equivalence.json`

The probe is strictly a persistence/reproducibility diagnostic and must not be interpreted as additional model-performance evidence.

## TEST governance

The Phase 5 TEST holdout is permanently consumed.

Phase 5G must not open the monitoring TEST dataset.

Phase 5G must not:

- generate new TEST predictions;
- generate new TEST performance metrics;
- retune from Phase 5F evidence;
- change threshold from Phase 5F evidence;
- change candidate from Phase 5F evidence;
- rerun Phase 5F.

The frozen Phase 5F result may be referenced as immutable provenance only.

## Scientific scope

Phase 5G preserves all previous limitations:

- synthetic data only;
- reduced-order argon monitoring environment;
- no experimental validation;
- no industrial validation;
- no OIPT operating-range claim;
- pressure-path diagnosis remains mechanism-ambiguous.

Persistence equivalence proves reproducibility of the frozen computational inference path. It does not establish physical accuracy, industrial readiness, experimental validity, or fault-diagnosis performance beyond the already-frozen Phase 5F evidence.

## Exit criteria

Phase 5G may close only when:

- this protocol was frozen before implementation;
- trusted manifest loading is implemented;
- six-to-nine feature construction is implemented;
- runtime prediction is implemented;
- frozen model bytes remain unchanged;
- production manifest is generated from a clean committed repository;
- artifact hash validation passes;
- exact runtime compatibility validation passes;
- deterministic persistence equivalence passes;
- full repository regression passes;
- Phase 5 TEST is not reopened;
- models are not retrained;
- models are not reserialized;
- models are not retuned.

Only after those conditions pass may Phase 5G be formally closed and Phase 5H unlocked.
