# Phase 5F — Locked TEST Evaluation Summary

## Status

Phase 5F is formally complete.

The one-time locked TEST evaluation was executed exactly once after:

- the Phase 5 protocol had been frozen;
- Phase 5E-R redevelopment had passed its validation gates;
- the detector and diagnoser had been selected using TRAIN/VALIDATION evidence only;
- the selected models had been refitted on TRAIN+VALIDATION;
- the final refit artifacts had been frozen;
- the dedicated Phase 5F evaluator had been implemented, synthetically qualified, and frozen.

The Phase 5 TEST holdout is now permanently consumed and must never be reused for development, model selection, threshold selection, retuning, or performance-improvement reruns.

## Frozen evaluation evidence

Result artifact:

`results/phase5/locked_test_evaluation.json`

SHA-256:

`63ff1ea3448e0ab49ef0ea13f3e4cfbdcea985b7ede624b0d61cb17d5da53fb3`

Evidence-freeze commit:

`3e6e357c57360c45e5a17a4fefde2a0c3c8da773`

Subject:

`data: freeze phase 5f locked test evidence`

## Final frozen models

### Fault detector

Candidate:

`ERD4_extra_trees_threshold_040`

Model family:

`ExtraTreesClassifier`

Final decision threshold:

`0.40`

Final fit rows:

`3072`

Training data:

TRAIN + VALIDATION only.

TEST fit rows:

`0`

### Fault diagnoser

Candidate:

`ERG3_extra_trees`

Model family:

`ExtraTreesClassifier`

Final fit rows:

`1451`

Training data:

Ground-truth active TRAIN + VALIDATION rows only after the frozen redevelopment class mapping.

Redeveloped diagnostic classes:

1. `flow_delivery`
2. `power_coupling`
3. `pressure_path_anomaly`

The `pressure_path_anomaly` class deliberately combines:

- `pressure_sensor_bias`
- `pumping_effectiveness`

This is an ambiguity-aware observable class. The available monitoring channels do not justify a robust claim that those two physical mechanisms can always be distinguished.

## TEST dataset

TEST rows:

`1024`

TEST episodes:

`16`

Fault episodes:

`12`

Normal episodes:

`4`

Active rows:

`463`

Inactive rows:

`561`

The TEST dataset is synthetic and derives from the frozen Phase 3 reduced-order monitoring environment.

## Detector TEST performance

Balanced accuracy:

`0.7998771862956846`

Macro F1:

`0.8038133073646447`

Active-fault recall:

`0.6781857451403888`

Positive precision:

`0.8770949720670391`

Specificity:

`0.9215686274509803`

False-positive rate:

`0.0784313725490196`

AUROC:

`0.8313063297182214`

Average precision:

`0.8682481767587944`

Confusion matrix:

| Truth / Prediction | Inactive | Active |
|---|---:|---:|
| Inactive | 517 | 44 |
| Active | 149 | 314 |

### Per-fault-family detector recall

| Original fault family | Recall |
|---|---:|
| flow_delivery | 0.7008547008547008 |
| power_coupling | 0.6310679611650486 |
| pressure_sensor_bias | 0.7142857142857143 |
| pumping_effectiveness | 0.6612903225806451 |

The detector therefore retained recall above `0.60` for every represented active fault family on TEST.

However, the overall active-fault recall of approximately `0.6782` is below the `0.75` validation acceptance threshold used during development.

This does not reopen model development. TEST is final generalisation evidence, not a new tuning or acceptance dataset.

## Sequence diagnostics

Fault episodes detected:

`12 / 12`

Fault-episode detection rate:

`1.0`

Mean detection delay:

`6.416666666666667` ordered steps

Median detection delay:

`4.5` ordered steps

Maximum detection delay:

`18` ordered steps

Missed fault episodes:

`0`

Fault episodes with a pre-active false alarm:

`6 / 12`

Normal episodes containing at least one false alarm:

`4 / 4`

The sequence unit is `step_index`.

It is not calibrated physical time and must not be described as seconds, milliseconds, or another physical time unit.

The sequence evidence shows that every TEST fault episode was eventually detected, but false-alarm behaviour remains a material limitation.

## Diagnoser TEST performance

Macro F1:

`0.7363247786819279`

Balanced accuracy:

`0.7531878405664814`

Minimum per-class recall:

`0.6666666666666666`

Majority-class baseline macro-F1 improvement:

`0.5068630223079902`

### Per-class diagnosis performance

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| flow_delivery | 0.5608108108108109 | 0.7094017094017094 | 0.6264150943396226 | 117 |
| power_coupling | 0.8425925925925926 | 0.883495145631068 | 0.8625592417061612 | 103 |
| pressure_path_anomaly | 0.782608695652174 | 0.6666666666666666 | 0.72 | 243 |

The redeveloped three-class diagnoser generalised materially better than the TRAIN-majority baseline and retained non-trivial recall for all three observable diagnostic classes.

## Secondary end-to-end TEST performance

The secondary end-to-end four-state system combines detection and diagnosis.

Classes:

1. `none`
2. `flow_delivery`
3. `power_coupling`
4. `pressure_path_anomaly`

Macro F1:

`0.6809791205369379`

Balanced accuracy:

`0.6683110939726208`

The end-to-end `pressure_path_anomaly` recall was approximately:

`0.41975308641975306`

This metric was explicitly frozen as report-only and was not used for candidate or threshold selection.

The lower end-to-end result reflects compounding detector and diagnoser errors and is an important limitation of the frozen monitoring system.

## Interpretation

The final TEST evidence supports the following limited conclusions:

- the detector provides useful discrimination on the frozen synthetic TEST holdout;
- detector specificity and positive precision are relatively strong;
- every TEST fault episode was eventually detected;
- the three-class redeveloped diagnoser generalises substantially better than a majority-class baseline;
- the ambiguity-aware pressure-path consolidation is materially more defensible than claiming reliable separation of pressure-sensor bias from pumping-effectiveness faults;
- end-to-end performance remains weaker than standalone diagnosis performance because detection errors propagate downstream.

The following limitations must be preserved:

- row-level active-fault recall is lower on TEST than the frozen development acceptance threshold;
- false alarms occurred before fault activation in multiple fault episodes;
- all four normal TEST episodes contained at least one false alarm;
- the pressure-path class remains a mechanism-ambiguous observable category;
- the monitoring environment is synthetic;
- the underlying process model is a reduced-order argon plasma model;
- no experimental validation has been performed;
- no industrial validation has been performed;
- no Oxford Instruments operating-range claim is made;
- ordered-step detection delay is not calibrated physical time.

## Governance

The Phase 5 TEST holdout was consumed exactly once.

After TEST consumption:

- model selection is prohibited;
- threshold selection is prohibited;
- retuning is prohibited;
- candidate-family changes are prohibited;
- TEST reruns for performance improvement are prohibited.

The frozen result must be treated as final generalisation evidence.

## Phase 5F closure decision

Phase 5F is CLOSED.

The computational and governance objectives of Phase 5F have been satisfied.

The final TEST evidence does not justify reopening development.

Phase 5G — Persistence & Reproducible Inference — may proceed using the already-frozen detector and diagnoser.

Phase 5G must not modify or retune the frozen models based on Phase 5F TEST results.
