# Phase 5G ? Persistence & Reproducible Inference Summary

## Status

**Phase 5G is CLOSED.**

Phase 5G completed the persistence and reproducible-inference layer for the frozen Phase 5 process-monitoring models.

**Phase 5H is UNLOCKED.**

Phase 5H ? Final Phase 5 Audit, Consolidation & Closure ? is the next controlled stage.

## Frozen protocol

The Phase 5G protocol was frozen before implementation.

Protocol:

`configs/phase5/phase5g_protocol.json`

SHA-256:

`2ca15b486bf15c3c439e19607812358f632151c94de15541ae6d27e04fe1574c`

Protocol-freeze commit:

`e89513a2b2ab748b7f54d03ba3b5b18b980d8376`

The protocol prohibited:

- model retraining;
- model reconstruction;
- model reserialization;
- threshold retuning;
- candidate reselection;
- reopening the Phase 5 TEST dataset;
- new TEST predictions;
- new TEST performance calculations.

## Frozen runtime implementation

Runtime implementation:

`src/plasma_ai/monitoring/phase5g_persistence.py`

SHA-256:

`7d8a91a6cb7b860bc62acad08a9b4624388106d4e009aff2eaef907c05057834`

Implementation-freeze commit:

`0ffba2d134f5b788957e30c2ec3c12f8f22920c2`

The runtime implementation provides:

- trusted Phase 5G protocol loading;
- exact SHA-256 verification before pickle deserialization;
- frozen model-wrapper validation;
- frozen estimator-parameter validation;
- exact scientific-runtime compatibility validation;
- six-raw-feature input validation;
- deterministic six-to-nine feature construction;
- detector active-probability inference;
- frozen threshold decision logic;
- detector-positive-only diagnosis;
- four-state end-to-end runtime output;
- production manifest generation;
- trusted production manifest loading.

## Frozen detector

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

The detector artifact remained byte-identical throughout Phase 5G.

## Frozen diagnoser

Artifact:

`artifacts/phase5/fault_diagnoser.pkl`

SHA-256:

`2e9dea016137594c707ecd3c32116691cc90d0127133f11c70372c93e6fcca84`

Candidate:

`ERG3_extra_trees`

Fit rows:

`1451`

Runtime diagnosis states are:

- `flow_delivery`
- `power_coupling`
- `pressure_path_anomaly`

`pressure_path_anomaly` remains an ambiguity-aware observable class and does not identify a unique physical mechanism.

## Runtime feature contract

The public runtime interface accepts exactly six raw inputs:

1. `nominal_absorbed_power_W`
2. `target_pressure_mTorr`
3. `nominal_flow_sccm`
4. `measured_absorbed_power_W`
5. `measured_flow_sccm`
6. `measured_pressure_mTorr`

The runtime constructs exactly three row-local residual features:

- absorbed-power relative residual;
- flow relative residual;
- pressure relative residual.

The final frozen model representation contains exactly nine features.

No history, future rows, target values, protected truth, simulator truth or TEST-derived features are introduced.

## Production monitoring manifest

Production manifest:

`artifacts/phase5/monitoring_manifest.json`

SHA-256:

`a18afb158aa7c02ac0e345b5f238a91e34f876d10dd43218c574554b874a9259`

Manifest-freeze commit:

`638326cf9e22de9bcf49b8017ba2fe33c93bdd27`

The manifest was generated from the clean committed runtime implementation.

The manifest records:

- frozen feature contracts;
- frozen prediction semantics;
- detector and diagnoser metadata;
- exact artifact hashes;
- source evidence;
- scientific-runtime versions;
- source Git commit;
- clean generation state;
- scientific-scope limitations.

The manifest is write-once and refuses overwrite.

## Persistence-equivalence evidence

Evidence:

`results/phase5/persistence_equivalence.json`

SHA-256:

`6f53ac1c9b919172bde8a2030ecbedaabab94f0aa5dbac3e6cd68f534cb1feb9`

Evidence-freeze commit:

`b287cba5dafbd772eea56a5d33c73cc103468c1a`

The persistence-equivalence diagnostic used a deterministic synthetic probe of exactly **45 rows**.

It used:

- no TRAIN data;
- no VALIDATION data;
- no TEST data;
- no target labels;
- no model-performance acceptance criteria.

The probe contained six raw runtime features and produced the frozen nine-feature model representation.

Exactly **36 probe rows** were detector-positive and therefore entered the diagnoser.

## Exact persistence-equivalence result

All frozen equivalence requirements passed:

- raw-to-model feature matrix: exact array equality;
- detector active probability: exact array equality;
- detector binary decision: exact array equality;
- diagnoser prediction: exact array equality;
- end-to-end runtime state: exact array equality;
- all probabilities finite.

Maximum absolute detector-probability difference:

`0.0`

No tolerance-based substitution was used.

This demonstrates exact computational reproducibility of the frozen inference path in the qualified runtime.

It is not new model-performance evidence.

## TEST governance

The Phase 5 TEST holdout was consumed exactly once during Phase 5F.

Phase 5G did not reopen the TEST dataset.

Phase 5G did not:

- generate new TEST predictions;
- calculate new TEST metrics;
- retune the detector;
- change the detector threshold;
- reselect a model;
- retrain either model;
- reconstruct either model;
- reserialize either model.

Frozen Phase 5F TEST evidence remains:

`results/phase5/locked_test_evaluation.json`

SHA-256:

`63ff1ea3448e0ab49ef0ea13f3e4cfbdcea985b7ede624b0d61cb17d5da53fb3`

## Scientific scope

Phase 5G remains inside the established project scope:

- synthetic data only;
- reduced-order argon process-monitoring environment;
- no experimental validation;
- no industrial validation;
- no OIPT operating-range claim;
- no claim that persistence equivalence establishes physical accuracy;
- no claim that persistence equivalence establishes industrial readiness;
- no claim that pressure-path anomaly identifies a unique mechanism.

## Repository qualification

Phase 5G targeted persistence/inference tests passed.

The full repository regression at Phase 5G closure is expected to remain:

`477 passed`

Frozen model artifacts, the Phase 5F TEST evidence, the Phase 5G protocol, production manifest and persistence-equivalence evidence remain immutable.

## Closure decision

All frozen Phase 5G exit requirements have been satisfied.

**Phase 5G is CLOSED.**

**Phase 5H is UNLOCKED.**

Phase 5H must perform the final Phase 5 audit, consolidate the complete monitoring evidence chain, confirm repository/governance integrity, formally close Phase 5, and only then unlock Phase 6.
