# Phase 4G ? Persistence, Provenance, Reproducible Inference and Speed Benchmark Summary

## Status

**Phase:** 4G
**Technical status:** FORMALLY CLOSED
**Formal closure:** COMPLETE
**Model redevelopment:** PROHIBITED
**TEST status:** CONSUMED - NOT ACCESSED IN PHASE 4G
**Phase 4H status:** UNLOCKED

---

## 1. Purpose

Phase 4G converted the frozen Phase 4 surrogate into a reproducibly loadable
inference artifact and benchmarked its inference runtime against the canonical
reduced-order source simulator.

The phase covered:

- persisted final surrogate models;
- metadata and configuration preservation;
- source and artifact provenance;
- deterministic model loading;
- reproducible physical-scale inference;
- persistence-equivalence validation;
- frozen inference-speed benchmarking;
- runtime evidence preservation.

Phase 4G was not a model-development phase.

No model selection, hyperparameter tuning, feature modification, target
transformation change, or TEST-based redevelopment was permitted.

---

## 2. Entry state

Phase 4G began only after Phase 4F had been formally closed.

Phase 4F formal closure commit:

`f146103`

Phase 4F frozen TEST artifact:

`results/phase4/locked_test_evaluation.json`

Phase 4F TEST artifact SHA-256:

`a14638013561b8f1f66547fab9910d96b0a2ad2c12afad894948b1f690af14e0`

The Phase 4 TEST split was already consumed before Phase 4G began.

Phase 4G did not reopen or reuse TEST targets.

---

## 3. Frozen Phase 4G protocol

Phase 4G protocol:

`docs/phase4/phase4g_protocol.md`

Machine-readable protocol:

`configs/phase4/phase4g_protocol.json`

Protocol freeze commit:

`771b16c`

The protocol was committed before any production inference timing result was
observed.

The frozen protocol declared:

- persistence format;
- artifact locations;
- loader-validation requirements;
- provenance requirements;
- persistence-equivalence rules;
- speed-benchmark point set;
- timing function;
- warm-up counts;
- repetition counts;
- workload definitions;
- summary statistics;
- speed-up definitions;
- scientific interpretation limits.

---

## 4. Frozen surrogate contract

Exact feature order:

1. `nominal_absorbed_power_W`
2. `target_pressure_mTorr`

Qualified numerical envelope:

- absorbed power: `15?90 W`;
- pressure: `10?60 mTorr`.

Final fit:

`TRAIN + VALIDATION = 6,144 rows`

No TEST row was used for Phase 4G fitting, persistence, benchmarking, or
inference validation.

---

## 5. Final density surrogate

Target:

`true_electron_density_m3`

Model:

`HistGradientBoostingRegressor`

Transform:

`log10`

Frozen configuration:

- learning rate: `0.05`;
- maximum iterations: `400`;
- maximum leaf nodes: `31`;
- L2 regularisation: `0.1`;
- monotonic constraints: `[1, 1]`;
- early stopping: `False`;
- random state: `20260913`.

Candidate identifier:

`phase4er_hist_gradient_boosting_lr0.05_iter400_leaves31_l20.1`

---

## 6. Final temperature surrogate

Target:

`true_electron_temperature_eV`

Model:

`ExtraTreesRegressor`

Transform:

`identity`

Frozen configuration:

- number of estimators: `500`;
- maximum depth: `None`;
- minimum samples per leaf: `1`;
- maximum features: `1.0`;
- `n_jobs`: `1`;
- random state: `20260913`.

Candidate identifier:

`extra_trees_n500_depthnone_leaf1_features1`

---

## 7. Persistence implementation

Persistence and reproducible-inference implementation:

`src/plasma_ai/surrogate/phase4g_persistence.py`

Implementation commit:

`669da54`

Targeted persistence regression:

`5 passed`

The implementation:

- reconstructs the already-frozen final surrogate;
- does not access TEST targets;
- validates the frozen protocol state;
- validates final model identities and parameters;
- serialises models using Python pickle protocol 5;
- records metadata and provenance;
- validates artifact hashes before deserialisation;
- validates runtime compatibility;
- validates feature ordering;
- validates physical target transforms;
- exposes deterministic physical-scale inference;
- refuses silent artifact overwrite.

---

## 8. Persistence format

Persisted artifacts:

`artifacts/phase4/density_model.pkl`

`artifacts/phase4/temperature_model.pkl`

Manifest:

`artifacts/phase4/surrogate_manifest.json`

Serialization:

- Python standard-library `pickle`;
- pickle protocol `5`;
- model loading restricted to trusted project artifacts;
- SHA-256 verified before deserialisation.

Pickle artifacts must not be loaded from untrusted or unauthenticated sources.

---

## 9. Persisted model hashes

Density model SHA-256:

`2706589929cdbebabda82cc9091ce110d673adc1c380bb49de9590f89da0b6a8`

Temperature model SHA-256:

`891c5f99082ae27f8dd08d610d528460ecae72e4d019c0bc47ba0f993420c7e5`

Persisted model sizes observed at generation:

- density: `1,031,084 bytes`;
- temperature: `442,206,424 bytes`.

The large temperature model is managed through Git LFS rather than ordinary
Git object storage.

---

## 10. Git LFS handling

Git LFS rule:

`artifacts/phase4/*.pkl`

Both production pickle artifacts are tracked through Git LFS.

Persisted-artifact freeze commit:

`803488d`

Git LFS does not alter the scientific contents or SHA-256 identities of the
persisted model files.

---

## 11. Manifest provenance

Production persistence was generated from clean committed source state.

Manifest source Git commit:

`669da540efe5a9632a3f195af11015da2e85cd61`

The manifest records, among other things:

- model identities;
- target names;
- transforms;
- feature names and ordering;
- units;
- qualified bounds;
- hyperparameters;
- training-data identity;
- final fit row count;
- artifact SHA-256 values;
- source Git state;
- Python and package environment;
- random states;
- scientific limitations.

---

## 12. Persistence equivalence

Persistence equivalence was evaluated on the already-frozen:

`41 x 41`

in-domain probe grid.

Total probe points:

`1,681`

Evidence artifact:

`results/phase4/persistence_equivalence.json`

Results:

### Density

- exact array equality: `True`;
- maximum absolute difference: `0.0`;
- maximum relative difference: `0.0`.

### Temperature

- exact array equality: `True`;
- maximum absolute difference: `0.0`;
- maximum relative difference: `0.0`.

Overall persistence-equivalence status:

`PASS`

Persisted and in-memory predictions were therefore bit-for-bit identical on
all 1,681 frozen probe points in the same runtime environment.

This is a reproducibility check, not experimental validation or OOD
validation.

---

## 13. Reproducible inference

The production loader:

- validates the manifest schema;
- validates protocol compatibility;
- validates feature ordering;
- validates model metadata;
- validates binary SHA-256 values;
- validates runtime compatibility;
- deserialises only after integrity checks;
- returns physical-scale predictions.

A final loader smoke test was performed at three in-domain operating points.

Results:

- density output shape valid;
- temperature output shape valid;
- all outputs finite;
- all density predictions positive;
- all temperature predictions positive.

Production loader smoke test:

`PASS`

---

## 14. Speed benchmark protocol

The speed benchmark was frozen before timing results were observed.

Benchmark implementation:

`src/plasma_ai/surrogate/phase4g_benchmark.py`

Benchmark implementation commit:

`e3fd699`

Targeted benchmark regression:

`7 passed`

Complete repository regression before benchmark execution:

`393 passed`

The working tree was clean before the real benchmark run.

---

## 15. Benchmark point set

The benchmark used an independent deterministic regular grid over the
qualified model envelope.

Grid:

`8 x 8`

Total operating points:

`64`

Bounds:

- absorbed power: `15?90 W`;
- pressure: `10?60 mTorr`.

The benchmark grid is independent of the TEST split.

No TEST target was accessed during benchmark execution.

---

## 16. Source workload

The source workload used the canonical:

`simulate_base_design_point`

path.

Timed operations included:

- canonical source operating-point construction;
- reduced-order plasma simulation;
- canonical qualification checks;
- physical source-target extraction.

Excluded operations included:

- configuration loading;
- benchmark-grid construction;
- result-file writing;
- JSON serialisation;
- console output.

Frozen timing configuration:

- warm-up workloads: `1`;
- timed repetitions: `3`;
- points per workload: `64`.

---

## 17. Scalar surrogate workload

Scalar surrogate inference evaluated the same 64 operating points one point at
a time.

Timed operations included:

- production input validation;
- density prediction;
- density inverse `log10` transformation;
- temperature prediction;
- physical-result construction.

Persisted model loading was excluded from timed inference.

Frozen timing configuration:

- warm-up workloads: `5`;
- timed repetitions: `200`;
- points per workload: `64`.

---

## 18. Batch surrogate workload

Batch surrogate inference evaluated the same 64 operating points in one
vectorized inference call.

Frozen timing configuration:

- warm-up workloads: `10`;
- timed repetitions: `1,000`;
- points per workload: `64`.

The workload included the same production prediction and physical-transform
operations as scalar inference.

Persisted model loading was excluded.

---

## 19. Timing method

Timing function:

`time.perf_counter_ns()`

Primary summary statistic:

`median`

For every workload the artifact preserves:

- all raw runtime observations;
- median;
- minimum;
- maximum;
- mean;
- population standard deviation;
- median runtime per point.

No fastest-run result was selected for the primary comparison.

---

## 20. Frozen speed result artifact

Artifact:

`results/phase4/inference_speed_benchmark.json`

SHA-256:

`989a3a7d87d3cc715454946ef385df6b271a60b3e1b696bc7dc8befb30ed3d8d`

Benchmark execution Git commit:

`e3fd699f01521c82e6bda62302c265e7c72d297b`

Benchmark-artifact freeze commit:

`547ecd9`

The benchmark artifact was validated mathematically before being committed.

---

## 21. Timing results

Median source runtime per point:

`209.4124203125 ms`

Median scalar-surrogate runtime per point:

`71.11772734375 ms`

Median batch-surrogate runtime per point:

`1.4026125 ms`

Speed-up relative to the canonical reduced-order source workload:

| Workload | Median time per point | Speed-up |
|---|---:|---:|
| Reduced-order source simulator | 209.4124203125 ms | 1.0x |
| Persisted surrogate ? scalar | 71.11772734375 ms | 2.944588193887268x |
| Persisted surrogate ? batch | 1.4026125 ms | 149.3016926004153x |

Primary Phase 4G speed results:

- scalar speed-up: `2.944588193887268x`;
- batch speed-up: `149.3016926004153x`.

---

## 22. Timing interpretation

The benchmark demonstrates that the frozen surrogate can execute faster than
the canonical reduced-order source simulation under the recorded hardware,
software, implementation, and workload conditions.

The scalar comparison represents one-point-at-a-time production inference.

The batch result demonstrates the much larger benefit available when the
surrogate is evaluated vectorially over multiple operating points.

The measured speed-ups are:

- hardware dependent;
- implementation dependent;
- workload dependent;
- Python/runtime dependent.

They are not industrial throughput guarantees.

They are not evidence of experimental plasma performance.

They are not model-selection evidence.

---

## 23. Final technical audit

The final Phase 4G technical audit passed.

Verified items included:

- persisted density SHA-256;
- persisted temperature SHA-256;
- manifest hash linkage;
- target identities;
- target transforms;
- persistence-equivalence status;
- exact 1,681-point reload equality;
- benchmark artifact SHA-256;
- benchmark Git provenance;
- frozen benchmark point count;
- frozen scalar speed-up;
- frozen batch speed-up;
- no industrial-throughput claim;
- no model-selection interpretation;
- no TEST evaluation;
- production loader smoke test.

Final audit result:

`PHASE4G_FINAL_TECHNICAL_AUDIT=PASS`

---

## 24. Regression evidence

Phase 4G targeted persistence tests:

`5 passed`

Phase 4G targeted benchmark tests:

`7 passed`

Complete repository regression after benchmark implementation:

`393 passed`

Complete repository regression remained:

`393 passed`

after the final benchmark artifact had been frozen.

---

## 25. TEST discipline

Phase 4G did not access TEST targets.

TEST status remains:

`CONSUMED`

The Phase 4F locked TEST artifact remains final Phase 4 surrogate
generalisation evidence.

Phase 4G results must not be used to reopen model development.

No additional Phase 4 TEST evaluation is permitted.

---

## 26. Scientific claim boundary

All Phase 4G interpretation remains subject to the existing project claim
boundary:

- the data are synthetic;
- the source is a reduced-order argon plasma model;
- the operating envelope is numerically qualified;
- the persisted surrogate approximates the reduced-order simulator;
- there is no experimental validation;
- there is no industrial validation;
- there is no OIPT operating-range claim;
- absorbed power is not generator RF power;
- reactive etch or deposition behaviour is not predicted;
- wafer-scale spatial behaviour is not modelled.

The 41 x 41 persistence-equivalence grid is an in-domain reproducibility
diagnostic.

The 8 x 8 timing grid is an in-domain runtime benchmark.

Neither is experimental or out-of-distribution validation.

---

## 27. Phase boundary

Phase 4G is formally closed.

All required Phase 4G evidence is complete:

- frozen Phase 4G protocol;
- persistence implementation;
- production persisted surrogate;
- manifest and provenance;
- Git LFS handling;
- deterministic production loader;
- reproducible physical-scale inference;
- exact 1,681-point persistence equivalence;
- frozen timing implementation;
- real 64-point speed benchmark;
- frozen benchmark artifact;
- final technical audit;
- closure-documentation factual QA;
- staged whitespace validation;
- closure-documentation commit;
- clean working-tree gate.

Closure evidence:

- Phase 4G protocol freeze commit: `771b16c`;
- persistence implementation commit: `669da54`;
- persisted-artifact freeze commit: `803488d`;
- benchmark implementation commit: `e3fd699`;
- benchmark-artifact freeze commit: `547ecd9`;
- closure-documentation commit: `fb740be`;
- closure-documentation factual QA: `105 / 105 checks passed`;
- final repository regression: `393 passed`;
- final technical audit: `PASS`;
- TEST targets accessed in Phase 4G: `False`.

The Phase 4 TEST split remains consumed and must not be reopened.

No surrogate redevelopment is permitted.

Phase 4H is formally unlocked for the final Phase 4 audit,
consolidation, and closure.
