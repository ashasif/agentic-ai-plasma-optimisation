# Phase 5H ? Final Phase 5 Audit, Consolidation & Closure

## Final status

**Phase 5 is CLOSED.**

**Phase 6 ? Constrained Operating-Point Optimisation ? is UNLOCKED.**

Phase 5 delivered the frozen process-monitoring, anomaly-detection and fault-diagnosis subsystem for the reduced-order synthetic argon plasma environment.

This final audit does not introduce new modelling, model selection, threshold selection, TEST evaluation or scientific claims.

## Phase 5 architecture

Phase 5 was executed as a controlled evidence chain:

- Phase 5A ? protocol and governance freeze;
- Phase 5B ? leakage-safe monitoring pipeline;
- Phase 5C ? healthy-reference anomaly monitoring;
- Phase 5D ? supervised fault detection and diagnosis benchmark;
- Phase 5E ? original validation selection and controlled stop;
- Phase 5E-R ? ambiguity-aware redevelopment;
- Phase 5F ? one-time locked TEST evaluation;
- Phase 5G ? persistence and reproducible inference;
- Phase 5H ? final audit, consolidation and closure.

## Monitoring data contract

The Phase 5 monitoring dataset originates from the frozen Phase 3 synthetic monitoring environment.

The approved public predictive channels are:

1. `nominal_absorbed_power_W`
2. `target_pressure_mTorr`
3. `nominal_flow_sccm`
4. `measured_absorbed_power_W`
5. `measured_flow_sccm`
6. `measured_pressure_mTorr`

Three deterministic row-local residuals are added:

- absorbed-power relative residual;
- flow relative residual;
- pressure relative residual.

The frozen predictive representation therefore contains nine features.

Episode identifiers, split labels and step indices are governance/order metadata rather than predictive features.

Protected simulator truth and fault-design variables are not predictive features.

## Fault structure and identifiability

The original synthetic fault families were:

- power coupling;
- flow delivery;
- pumping effectiveness;
- pressure sensor bias.

Development evidence showed that pumping effectiveness and pressure sensor bias were not sufficiently identifiable as separate mechanisms using the approved monitoring channels.

The frozen ambiguity-aware diagnosis therefore uses three active classes:

- `flow_delivery`
- `power_coupling`
- `pressure_path_anomaly`

`pressure_path_anomaly` combines the original pumping-effectiveness and pressure-sensor-bias families for diagnosis.

This class indicates an observable pressure-path anomaly. It does not establish the unique underlying physical mechanism.

## Original controlled stop

The original Phase 5D/5E supervised detector/diagnoser development did not satisfy the frozen validation acceptance requirements.

The project therefore executed a controlled stop before TEST rather than weakening acceptance criteria after observing the result.

The original failure evidence remains frozen and preserved.

## Phase 5E-R redevelopment

Redevelopment was protocol-controlled and validation-only.

The frozen detector is:

`ERD4_extra_trees_threshold_040`

with decision threshold:

`0.40`

The final detector was refitted on TRAIN + VALIDATION only:

`3072` rows.

The frozen diagnoser is:

`ERG3_extra_trees`

The final diagnoser was refitted on mapped active TRAIN + VALIDATION rows only:

`1451` rows.

No TEST rows were used for fitting.

Frozen artifacts:

`artifacts/phase5/fault_detector.pkl`

SHA-256:

`4a00e785d7e7f0a8429020399967cdd82c18da275d58e5b83bc11707eeac8a6a`

and:

`artifacts/phase5/fault_diagnoser.pkl`

SHA-256:

`2e9dea016137594c707ecd3c32116691cc90d0127133f11c70372c93e6fcca84`

## One-time locked TEST evaluation

Phase 5 TEST was consumed exactly once during Phase 5F.

Frozen TEST evidence:

`results/phase5/locked_test_evaluation.json`

SHA-256:

`63ff1ea3448e0ab49ef0ea13f3e4cfbdcea985b7ede624b0d61cb17d5da53fb3`

No post-TEST retuning, candidate selection or threshold selection is permitted.

### Detector TEST evidence

Final frozen detector TEST metrics include:

- balanced accuracy: `0.7998771862956846`;
- macro F1: `0.8038133073646447`;
- active-fault recall: `0.6781857451403888`;
- positive precision: `0.8770949720670391`;
- specificity: `0.9215686274509803`;
- false-positive rate: `0.0784313725490196`;
- AUROC: `0.8313063297182214`;
- average precision: `0.8682481767587944`.

Active-fault recall on TEST is below the development acceptance target of `0.75`.

This is preserved as final holdout evidence and is not used to retune the frozen detector.

Per-original-family detector recall remained above the frozen `0.60` development floor for each synthetic fault family.

### Sequence-level evidence

All `12` synthetic fault episodes were eventually detected.

Detection delay in `step_index` units:

- mean: `6.416666666666667`;
- median: `4.5`;
- maximum: `18`.

`step_index` is ordering metadata and is not calibrated physical time.

False-alarm behaviour remains a material limitation:

- `6` fault episodes contained pre-active false alarms;
- all `4` normal TEST episodes contained at least one false alarm.

### Diagnoser TEST evidence

Ground-truth-active-row diagnosis produced:

- macro F1: `0.7363247786819279`;
- balanced accuracy: `0.7531878405664814`;
- minimum class recall: `0.6666666666666666`.

The TRAIN-majority diagnostic baseline macro F1 was:

`0.2294617563739377`

and the frozen macro-F1 improvement over that baseline was:

`0.5068630223079902`.

These are final TEST observations, not post-TEST selection criteria.

### End-to-end monitoring evidence

The secondary four-state end-to-end result reported:

- macro F1: `0.6809791205369379`;
- balanced accuracy: `0.6683110939726208`.

Frozen class recall:

- `none`: `0.9215686274509803`;
- `flow_delivery`: `0.7008547008547008`;
- `power_coupling`: `0.6310679611650486`;
- `pressure_path_anomaly`: `0.41975308641975306`.

The relatively weak end-to-end pressure-path recall is an explicit limitation.

This four-state result is report-only and was not a model-selection metric.

## Persistence and reproducible inference

Phase 5G added the trusted inference layer without retraining or reserializing the frozen models.

Frozen production manifest:

`artifacts/phase5/monitoring_manifest.json`

SHA-256:

`a18afb158aa7c02ac0e345b5f238a91e34f876d10dd43218c574554b874a9259`

The trusted loader verifies artifact hashes before pickle deserialization and validates frozen wrapper, parameter and runtime contracts.

The public runtime interface accepts the six raw monitoring channels and deterministically constructs the nine-feature model representation.

The detector threshold remains exactly `0.40`.

The diagnoser runs only for detector-positive rows.

Runtime end-to-end states are:

- `none`
- `flow_delivery`
- `power_coupling`
- `pressure_path_anomaly`

## Persistence equivalence

Frozen persistence-equivalence evidence:

`results/phase5/persistence_equivalence.json`

SHA-256:

`6f53ac1c9b919172bde8a2030ecbedaabab94f0aa5dbac3e6cd68f534cb1feb9`

The reproducibility diagnostic used exactly `45` deterministic synthetic probe rows.

It used:

- no TRAIN data;
- no VALIDATION data;
- no TEST data;
- no target labels;
- no model-performance acceptance criterion.

Exact equality passed for:

- six-to-nine feature construction;
- detector probabilities;
- detector decisions;
- diagnoser predictions;
- end-to-end monitoring states.

Maximum detector-probability difference was:

`0.0`

This establishes reproducibility of the frozen inference path in the qualified runtime.

It is not model-performance validation, OOD validation, experimental validation or industrial validation.

## Governance closure

The Phase 5 evidence chain preserves the original failure, the controlled redevelopment decision, the one-time TEST evaluation and the final deployment-oriented runtime evidence.

The TEST holdout is permanently consumed.

Phase 5 TEST must not be rerun for development.

The frozen models must not be retuned from TEST evidence.

The detector threshold must not be altered from TEST evidence.

The ambiguity-aware class mapping must not be silently replaced with mechanism-specific claims.

## Scientific limitations

There is no experimental validation.

There is no industrial validation.

Phase 5 uses synthetic data generated from the reduced-order project environment.

Phase 5 does not establish:

- experimental plasma-fault validation;
- industrial process-monitoring validation;
- OIPT hardware performance;
- an OIPT operating range;
- wafer-scale spatial plasma diagnosis;
- reactive etch/deposition chemistry diagnosis;
- physical-time-calibrated detection delay;
- unique mechanism identification for pressure-path anomalies.

These limitations remain binding for downstream phases.

## Phase 6 interface

Phase 6 is now permitted to begin.

Phase 6 is:

**Constrained Operating-Point Optimisation**

Phase 6 should use the frozen upstream scientific contracts and must not silently reopen or retune Phase 5.

The frozen Phase 4 surrogate remains the principal modelling input for constrained operating-point optimisation.

Phase 5 monitoring/diagnostic outputs may be used only where the Phase 6 protocol explicitly defines a justified interface.

Agentic workflow orchestration remains reserved for Phase 7.

## Final closure decision

The Phase 5 protocol, development evidence, redevelopment evidence, frozen models, one-time locked TEST evidence, persistence manifest, persistence-equivalence evidence and closure documentation have been audited.

The repository regression passes and the evidence chain remains immutable.

**Phase 5 is formally CLOSED.**

**Phase 6 is formally UNLOCKED.**
