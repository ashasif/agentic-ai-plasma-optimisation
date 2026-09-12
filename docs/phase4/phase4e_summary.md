# Phase 4E - Physics-Aware Pre-Test Acceptance

**Status:** CONTROLLED STOP - DENSITY ACCEPTANCE FAILED
**Phase 4F status:** BLOCKED
**Final TRAIN + VALIDATION refit:** NOT PERFORMED

## 1. Scope

Phase 4E performed the frozen pre-test physics-aware acceptance procedure for
the two validation-selected surrogate candidates produced by Phase 4D.

The phase used:

- the frozen Phase 4 feature contract;
- the frozen Phase 4 target contract;
- the Phase 4D-selected model configurations;
- TRAIN only for candidate fitting during acceptance;
- the deterministic 41 x 41 in-envelope physics-validation probe grid;
- the canonical Phase 3 reduced-order source simulator as the structural
  reference;
- the pre-approved Phase 4E acceptance amendment.

TEST target values were not accessed.

No Phase 4F locked TEST evaluation was performed.

No final TRAIN + VALIDATION refit was performed because the Phase 4E acceptance
gate did not pass.

No production model was persisted.

## 2. Frozen decision semantics

The Phase 4A protocol defined the physics-aware checks but did not originally
assign deterministic numerical PASS/FAIL semantics to unsupported monotonic
trend violations or local spurious oscillation.

Before any Phase 4E probe-grid results were inspected, the following amendment
was approved and frozen:

`docs/phase4/phase4e_acceptance_amendment.md`

The frozen acceptance rules require:

- finite and strictly positive surrogate predictions;
- zero source-unsupported adjacent trend reversals;
- zero source-unsupported strict local turning points;
- both target-specific candidates to pass before final refitting is permitted.

There is no post-result tolerance adjustment.

A failed candidate may not be replaced silently by another Phase 4D candidate.

No retuning, transform modification, feature modification, model reselection,
or acceptance-rule alteration is permitted in response to the observed Phase 4E
result.

## 3. Probe-grid design

The physics-validation grid is the frozen regular Cartesian design:

- absorbed power: 41 equally spaced values from 15 W to 90 W inclusive;
- target pressure: 41 equally spaced values from 10 mTorr to 60 mTorr
  inclusive;
- total probe points: 1,681;
- unique probe points: 1,681.

The feature order is:

1. `nominal_absorbed_power_W`
2. `target_pressure_mTorr`

The probe grid lies inside the numerically qualified source-model envelope.

It is an in-envelope interpolation/behavioural diagnostic grid.

It is not an OOD test set.

Probe points were not appended to TRAIN, VALIDATION, or TEST.

## 4. Canonical source reference

The Phase 3 reduced-order simulator was evaluated on the full 1,681-point probe
grid before either surrogate was evaluated.

Source-reference gate:

- total points: 1,681;
- unique points: 1,681;
- finite source targets: PASS;
- integration-successful rows: 1,681 / 1,681;
- converged rows: 1,681 / 1,681;
- qualification-valid rows: 1,681 / 1,681;
- model-valid rows: 1,681 / 1,681.

Domain-status counts:

- `supported`: 1,225;
- `near_boundary`: 456.

Source target ranges:

- electron density:
  `1.653204965862e+16` to `2.251090635821e+17 m^-3`;
- electron temperature:
  `1.547275635012` to `2.074523104787 eV`.

Frozen source-reference numerical-array SHA-256:

`4c6af7001e869f35572d451e293c92a1950f1a32c10a29134401a3845380d77a`

Frozen source-reference artifact:

`results/phase4/source_reference_grid.json`

Frozen source-reference artifact file SHA-256:

`a7a76e4d14e470e5458bbcbe8a5537d0db1dade9693c8eb917b382003480673c`

The full source reference was independently regenerated and reproduced the
same numerical-array SHA-256 before being frozen.

## 5. Source structural behaviour

The source simulator's own grid behaviour was characterised before surrogate
acceptance.

### Electron density versus absorbed power at fixed pressure

Across all adjacent grid intervals:

- positive increments: 1,640;
- negative increments: 0;
- zero increments: 0;
- strict turning points: 0.

The source density is therefore strictly increasing with absorbed power on
every frozen fixed-pressure diagnostic line.

### Electron density versus pressure at fixed absorbed power

Across all adjacent grid intervals:

- positive increments: 1,640;
- negative increments: 0;
- zero increments: 0;
- strict turning points: 0.

The source density is therefore strictly increasing with pressure on every
frozen fixed-power diagnostic line.

### Electron temperature versus pressure at fixed absorbed power

Across all adjacent grid intervals:

- positive increments: 0;
- negative increments: 1,640;
- zero increments: 0;
- strict turning points: 0.

The source electron temperature is therefore strictly decreasing with pressure
on every frozen fixed-power diagnostic line.

These are measured properties of this specific reduced-order source simulator.

They are not asserted as universal experimental ICP laws.

## 6. Phase 4D-selected density candidate

Target:

`true_electron_density_m3`

Selected target transformation:

`log10`

Selected model:

`ExtraTreesRegressor`

Frozen candidate ID:

`extra_trees_n500_depth16_leaf1_features1`

Frozen parameters:

- `n_estimators = 500`
- `max_depth = 16`
- `min_samples_leaf = 1`
- `max_features = 1.0`
- `random_state = 20260913`
- `n_jobs = 1`

For Phase 4E acceptance the candidate was rebuilt from the frozen Phase 4D
selection artifact and fitted using TRAIN only:

- fit rows: 4,096;
- VALIDATION used for fitting: no;
- VALIDATION used for Phase 4E acceptance: no;
- TEST targets accessed: no.

Physical-scale probe predictions:

- minimum:
  `1.800881667839e+16 m^-3`;
- maximum:
  `2.224459641917e+17 m^-3`.

### Density positivity

- non-finite predictions: 0;
- non-positive predictions: 0;
- positivity: PASS.

### Density versus absorbed power

- unsupported trend reversals: 0;
- source-unsupported strict turning points: 0;
- result: PASS.

### Density versus pressure

- unsupported trend reversals: 1;
- source-unsupported strict turning points: 1;
- result: FAIL.

Both violations localise to grid location:

`[2, 39]`

This corresponds to:

- absorbed power: `18.75 W`;
- turning-point pressure: `58.75 mTorr`;
- adjacent pressure interval producing the reversal:
  `58.75 -> 60.00 mTorr`.

The source values were:

- at `57.50 mTorr`:
  `4.613795257735e+16 m^-3`;
- at `58.75 mTorr`:
  `4.652151465889e+16 m^-3`;
- at `60.00 mTorr`:
  `4.689774990499e+16 m^-3`.

The corresponding source increments were:

- left increment:
  `+3.835620815437e+14 m^-3`;
- right increment:
  `+3.762352461008e+14 m^-3`.

The surrogate values were:

- at `57.50 mTorr`:
  `4.624639506396e+16 m^-3`;
- at `58.75 mTorr`:
  `4.641404194740e+16 m^-3`;
- at `60.00 mTorr`:
  `4.635257758945e+16 m^-3`.

The corresponding surrogate increments were:

- left increment:
  `+1.676468834432e+14 m^-3`;
- right increment:
  `-6.146435795341e+13 m^-3`.

The right surrogate increment is therefore opposite in direction to the
positive source increment.

Its signed magnitude relative to the source increment is:

`-0.1633668259165`

The surrogate creates a strict local turning point because the adjacent
surrogate increments have opposite signs.

Pointwise relative errors remained numerically modest:

- at `58.75 mTorr`: approximately `0.2310%`;
- at `60.00 mTorr`: approximately `1.1625%`.

These pointwise errors do not override the pre-approved structural acceptance
rule.

### Density decision

**FAIL**

The density candidate violates the frozen zero-tolerance rule for:

- unsupported trend reversals;
- source-unsupported strict local turning points.

## 7. Phase 4D-selected temperature candidate

Target:

`true_electron_temperature_eV`

Selected target transformation:

`identity`

Selected model:

`ExtraTreesRegressor`

Frozen candidate ID:

`extra_trees_n500_depthnone_leaf1_features1`

Frozen parameters:

- `n_estimators = 500`
- `max_depth = None`
- `min_samples_leaf = 1`
- `max_features = 1.0`
- `random_state = 20260913`
- `n_jobs = 1`

For Phase 4E acceptance the candidate was rebuilt from the frozen Phase 4D
selection artifact and fitted using TRAIN only.

Physical-scale probe predictions:

- minimum:
  `1.547320232327 eV`;
- maximum:
  `2.074477760193 eV`.

### Temperature positivity

- non-finite predictions: 0;
- non-positive predictions: 0;
- result: PASS.

### Temperature versus pressure

- unsupported trend reversals: 0;
- source-unsupported strict turning points: 0;
- result: PASS.

### Temperature decision

**PASS**

## 8. Overall Phase 4E decision

Target-specific decisions:

- density: FAIL;
- temperature: PASS.

Frozen overall rule:

Both candidates must pass.

Observed overall decision:

**STOP**

Machine-readable acceptance artifact:

`results/phase4/physics_acceptance.json`

Frozen artifact file SHA-256:

`27a01fb558b893a55744bb0ca053f24700c3a66bd980de894db6c588182f47d3`

The acceptance result was reproduced before the artifact was frozen.

## 9. Consequences of the STOP decision

Because the density candidate failed Phase 4E:

- final TRAIN + VALIDATION refit is not permitted;
- the 6,144-row refit was not performed;
- Phase 4F locked TEST evaluation is not permitted;
- TEST targets remain locked;
- the second-best Phase 4D model was not substituted;
- no model was retuned;
- no hyperparameter was changed;
- the density target transformation was not changed;
- the feature set was not changed;
- the 2% validation-selection rule was not changed;
- the qualified numerical envelope was not changed;
- the probe grid was not changed;
- the acceptance criteria were not changed;
- no final production surrogate was persisted;
- no final inference-speed benchmark was performed.

The STOP therefore preserves the frozen experimental protocol rather than
optimising the procedure after observing an unfavourable result.

## 10. Interpretation

The Phase 4D validation-selected density surrogate remains statistically strong,
but it does not satisfy the stricter source-referenced structural acceptance
criterion at one local high-pressure interval near the low-power edge of the
qualified model envelope.

The failure is highly localised rather than a broad loss of predictive
agreement.

Nevertheless, under the pre-approved zero-violation rule, a single unsupported
reversal is sufficient to fail the candidate.

The correct result is therefore not to reinterpret the violation as negligible
after inspection.

It must be recorded as a Phase 4E acceptance failure.

Any future attempt to continue surrogate development would require an explicitly
separate experimental iteration with its own predeclared protocol and without
using the locked TEST set as development information.

## 11. TEST-lock status

Throughout Phase 4E:

- default dataset loading withheld TEST targets;
- TEST features were permitted only for schema/split verification;
- `include_test_targets=True` was not used;
- no TEST metrics were calculated;
- no TEST result influenced model selection or acceptance.

TEST therefore remains unopened for target evaluation.

## 12. Reproducibility and implementation

Phase 4E added reusable implementation for:

- deterministic frozen probe-grid construction;
- canonical source-reference evaluation;
- source-reference artifact generation;
- deterministic source-reference hashing;
- exact Phase 4D winner reconstruction;
- TRAIN-only candidate fitting;
- physical-scale prediction after target-transform inversion;
- strict positivity checks;
- source-referenced trend comparison;
- strict source-referenced local-turning-point checks;
- deterministic PASS/FAIL aggregation;
- structured acceptance-result writing;
- TEST-lock enforcement;
- automated regression tests.

Key Phase 4E Git checkpoints include:

- `0f9c2de` - freeze Phase 4E pre-test acceptance rules;
- `8ab2d4d` - add frozen Phase 4E physics probe grid;
- `f3ae128` - add Phase 4E source reference evaluator;
- `4654a52` - add Phase 4E source reference artifact;
- `5e850a2` - freeze Phase 4E source reference grid;
- `85cdd95` - add Phase 4E physics acceptance logic;
- `49f1089` - add Phase 4E selected candidate fitting;
- `cde8bce` - add Phase 4E physics acceptance experiment;
- `214e524` - add Phase 4E acceptance result writer;
- `dbe645e` - freeze Phase 4E physics acceptance result.

## 13. Scientific claim boundary

Phase 4E preserves the following limitations:

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

The project demonstrates transferable research-engineering capability in
physics-aware surrogate modelling and validation.

It does not claim reproduction of Oxford Instruments Plasma Technology's
industrial process-modelling system.

## 14. Phase boundary after Phase 4E

Phase 4E reached a deterministic protocol-governed STOP.

It has not performed:

- final TRAIN + VALIDATION refitting;
- Phase 4F locked TEST evaluation;
- Phase 4G production persistence;
- Phase 4G speed benchmarking;
- Phase 4H formal Phase 4 closure.

Phase 4F is blocked under the current frozen Phase 4 experimental iteration.

Phase 4E closure QA completed:

- technical-summary factual QA: PASS, 30 / 30 checks;
- targeted Phase 4E regression suite: PASS, 47 / 47 tests;
- complete repository regression suite: PASS, 322 / 322 tests;
- full regression runtime: 73.36 s;
- TEST targets remained locked;
- the controlled STOP decision remained unchanged.

Remaining formal closure actions are:

1. stage the Phase 4E closure documentation;
2. pass `git diff --cached --check`;
3. create the Phase 4E closure commit;
4. confirm a clean working tree;
5. formally declare Phase 4E closed as a controlled STOP.
