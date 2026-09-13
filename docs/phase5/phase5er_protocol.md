# Phase 5E-R ? Validation-Only Redevelopment for Observable Fault Diagnosis

## Status

**Protocol state:** FROZEN content pending Git freeze commit
**Parent controlled-stop commit:** `f08bb6ddc340066cc857fb615e6f414bab13c596`
**Phase 5F:** LOCKED
**TEST access during redevelopment:** FORBIDDEN

## 1. Why redevelopment is required

The original Phase 5E validation path correctly stopped before TEST because
zero detector candidates and zero four-family diagnoser candidates satisfied
all frozen acceptance criteria.

That result remains frozen and is not overwritten by this redevelopment.

Validation has now been used as explicit development feedback. It must not be
described as an unseen final holdout. The locked Phase 5 TEST remains
unconsumed for predictive evaluation.

## 2. Source-level observability finding

The Phase 3 generator implements `pumping_effectiveness` as a process-layer
change to pumping effectiveness before the reduced-order process solve.

`pressure_sensor_bias` instead acts at the observation layer after the
process solve and modifies the measured-pressure signal.

The approved predictive contract contains measured pressure but no independent
measurement of true pressure, pumping speed, electron density, electron
temperature, or another plasma-state diagnostic.

Therefore the current observables do not support a robust mechanism-level
claim distinguishing pumping degradation from pressure-sensor bias.

The mechanisms are **not claimed to be algebraically identical**. Their
severity distributions and operating-point responses may differ. The issue is
that exploiting those dataset-specific differences would not establish robust
physical mechanism identification.

## 3. Redeveloped diagnostic resolution

The active diagnostic target becomes:

1. `flow_delivery`
2. `power_coupling`
3. `pressure_path_anomaly`

Mapping:

- `flow_delivery` -> `flow_delivery`
- `power_coupling` -> `power_coupling`
- `pressure_sensor_bias` -> `pressure_path_anomaly`
- `pumping_effectiveness` -> `pressure_path_anomaly`

A `pressure_path_anomaly` result means that the pressure observation path is
abnormal. It must **not** be reported as proof of either a sensor fault or a
pumping fault individually.

## 4. Predictive feature contract

The original nine Phase 5 model features remain unchanged:

- `nominal_absorbed_power_W`
- `target_pressure_mTorr`
- `nominal_flow_sccm`
- `measured_absorbed_power_W`
- `measured_flow_sccm`
- `measured_pressure_mTorr`
- `absorbed_power_relative_residual`
- `flow_relative_residual`
- `pressure_relative_residual`

No protected truth is admitted.

No Phase 4 surrogate outputs are introduced.

No synthetic electron-density or electron-temperature measurements are
fabricated.

No future or centred temporal features are permitted.

`step_index`, `episode_id`, and `split` remain ordering/grouping metadata only.

## 5. Detector redevelopment

Only the two strongest original detector model families are retained.

Their fitted-model hyperparameters remain unchanged.

The only redevelopment variable is the binary decision threshold.

Frozen candidates:

- ERD1 ? HistGradientBoosting, threshold 0.45
- ERD2 ? HistGradientBoosting, threshold 0.40
- ERD3 ? ExtraTrees, threshold 0.45
- ERD4 ? ExtraTrees, threshold 0.40

The detector acceptance criteria remain exactly unchanged from the original
Phase 5 protocol:

- balanced accuracy >= 0.75
- macro F1 >= 0.75
- active-fault recall >= 0.75
- specificity >= 0.75
- every represented active fault-family detection recall >= 0.60

No threshold may be introduced after redevelopment results are observed.

## 6. Diagnoser redevelopment

The original Logistic Regression, HistGradientBoosting and ExtraTrees
classifier families are retained with their original hyperparameters.

Because the diagnostic task is reduced from four classes to three, the
acceptance gate is deliberately strengthened:

- macro F1 >= 0.60
- balanced accuracy >= 0.60
- every class recall >= 0.50
- macro-F1 improvement over TRAIN-majority baseline >= 0.10

This is not a relaxation of the failed original gate.

## 7. Selection

Detector selection is permitted only among candidates passing every frozen
detector acceptance criterion.

Diagnoser selection is permitted only among candidates passing every frozen
redevelopment diagnosis criterion.

If either task has zero passing candidates, redevelopment stops before TEST.

## 8. Final refit

Only after both tasks pass validation acceptance:

- detector: TRAIN + VALIDATION, 3072 rows;
- diagnoser: active TRAIN + VALIDATION, 1451 rows;
- TEST fitting rows: 0.

No model-family, threshold, target mapping, preprocessing or feature change is
allowed after selection.

## 9. TEST discipline

Phase 5 TEST remains locked throughout Phase 5E-R.

No TEST feature values, targets, predictions or performance may be used for:

- candidate development;
- threshold choice;
- model selection;
- target-definition changes;
- acceptance changes.

Phase 5F may unlock only after passing redevelopment acceptance, recorded
selection, final-refit evidence, and a clean repository checkpoint.

## 10. Scientific claim boundary

All evidence remains synthetic.

The redevelopment does not restore four-mechanism fault identifiability.

`pressure_path_anomaly` is intentionally an ambiguity-aware diagnosis.

No experimental validation, industrial validation, or OIPT operating-range
claim is permitted.

## 11. Provenance

- Original Phase 5 protocol SHA256:
  `70de30b7a5f2e36b567f8848ee0e626833c25eb8285e7b4df9e63d5db77ddcec`
- Frozen Phase 5D benchmark SHA256:
  `fea87ce628e20add11044bca17323143b2b9bb436d3d4192ad94cac6d00856d4`
- Frozen controlled-stop record SHA256:
  `0f44f13a1efa4d95f8e70642ed129618616aec7a3b03b1dbe42e9a8a3ea60916`
- Phase 3 feature manifest SHA256:
  `ece31a97b317bda2da3a366dec1fb69d8756e05f3da64b10268d033dabbe948b`
- Monitoring process source SHA256:
  `a4f345b13a19dd051d3209cd53238dcd5f69a5285778ee20aea2fa89a0000738`
- Monitoring observation source SHA256:
  `7aca13e5a72171566479eac4cfde01a11d39c28049218f3edb2abd322dd06a8d`

No redevelopment modelling is authorised until this protocol is frozen in Git.
