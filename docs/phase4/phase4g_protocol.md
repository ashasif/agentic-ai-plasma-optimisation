# Phase 4G - Persistence, Provenance, Reproducible Inference and Speed Benchmark Protocol

## Status

**Protocol state:** FROZEN BEFORE BENCHMARK EXECUTION
**Phase:** 4G
**Purpose:** Persist the already-frozen final Phase 4 surrogate, provide reproducible loading and inference, record auditable provenance, and benchmark inference speed against the canonical reduced-order source simulator.

Phase 4F was formally closed at Git commit:

`f146103`

At entry to Phase 4G:

- the final surrogate configuration is frozen;
- TRAIN + VALIDATION final fitting contains exactly 6,144 rows;
- TEST has already been consumed exactly once in Phase 4F;
- the TEST result must not be rerun or used for development;
- Phase 4G must not perform model selection, retuning, feature changes, target-transform changes, or acceptance-rule changes.

No timing result may be generated before this protocol and its machine-readable benchmark specification are frozen in Git.

---

## 1. Frozen final surrogate

### 1.1 Feature contract

Exact feature order:

1. `nominal_absorbed_power_W`
2. `target_pressure_mTorr`

Feature units:

- nominal absorbed power: `W`
- target pressure: `mTorr`

Qualified numerical envelope:

- absorbed power: 15-90 W
- target pressure: 10-60 mTorr

No automatic feature discovery is permitted.

### 1.2 Density surrogate

Target:

`true_electron_density_m3`

Target unit:

`m^-3`

Estimator:

`HistGradientBoostingRegressor`

Target transform:

`log10`

Frozen configuration:

- `learning_rate = 0.05`
- `max_iter = 400`
- `max_leaf_nodes = 31`
- `l2_regularization = 0.1`
- `monotonic_cst = [1, 1]`
- `early_stopping = False`
- `random_state = 20260913`

Frozen candidate ID:

`phase4er_hist_gradient_boosting_lr0.05_iter400_leaves31_l20.1`

### 1.3 Temperature surrogate

Target:

`true_electron_temperature_eV`

Target unit:

`eV`

Estimator:

`ExtraTreesRegressor`

Target transform:

`identity`

Frozen configuration:

- `n_estimators = 500`
- `max_depth = None`
- `min_samples_leaf = 1`
- `max_features = 1.0`
- `random_state = 20260913`
- `n_jobs = 1`

Frozen candidate ID:

`extra_trees_n500_depthnone_leaf1_features1`

---

## 2. Frozen source evidence

Primary frozen Phase 3 dataset:

`data/synthetic/phase3/base_steady_state.csv`

SHA-256:

`dc8d3a24f84b3eca9ce930a88e969e51892d69430ce5840cc546523e9f9dd423`

Rows:

8,192

Final fitting rows:

6,144

Frozen Phase 4E-R final-refit evidence:

`results/phase4/phase4er_final_refit.json`

SHA-256:

`4132ff5221c14cf6753e1a19bed5ecaa17577462fcecce1699b57c719a81aa12`

Frozen Phase 4F TEST evidence:

`results/phase4/locked_test_evaluation.json`

SHA-256:

`a14638013561b8f1f66547fab9910d96b0a2ad2c12afad894948b1f690af14e0`

The Phase 4F evaluator must not be executed during Phase 4G.

---

## 3. Persistence design

Phase 4G will persist the two already-frozen fitted estimators as binary model artifacts under:

`artifacts/phase4/`

A separate machine-readable JSON manifest will describe and authenticate the persisted surrogate.

The persistence implementation must not introduce a new third-party serialization dependency.

The binary model representation will therefore use the Python standard-library `pickle` module with an explicitly fixed pickle protocol rather than relying on an undeclared direct dependency.

The persisted deliverable will contain, at minimum:

- the fitted density estimator;
- the fitted temperature estimator;
- target-transform identity required for physical-scale inference;
- exact feature ordering;
- target identities;
- artifact hashes;
- metadata required to reconstruct the meaning of each estimator.

Persisted artifacts are trusted project artifacts only.

Because pickle-based formats can execute code during loading, the loader must not be presented as safe for arbitrary or untrusted model files. The production loader must validate the expected repository manifest and artifact SHA-256 before deserializing model binaries.

---

## 4. Provenance contract

The Phase 4G manifest must record at minimum:

- artifact/schema version;
- model type for each target;
- candidate ID for each target;
- target name;
- target unit;
- feature names in exact frozen order;
- feature units;
- target transformation;
- exact estimator hyperparameters;
- explicit structural parameters not contained in the Phase 4 search dictionaries;
- random state where applicable;
- final fitting split identity;
- final fitting row count;
- Phase 3 dataset path;
- Phase 3 dataset SHA-256;
- Phase 4E-R final-refit artifact path and SHA-256;
- Phase 4F locked TEST artifact path and SHA-256;
- source Git commit;
- repository dirty/clean status at artifact generation;
- Python version;
- NumPy version;
- SciPy version;
- scikit-learn version;
- platform information;
- CPU information;
- training/reconstruction timestamp in UTC;
- artifact creation timestamp in UTC;
- qualified input bounds;
- scientific limitations;
- SHA-256 of each persisted model binary.

The source Git commit stored in the production manifest must identify the committed code used to create the persisted artifacts.

Persisted production artifacts must not be generated from an uncommitted persistence implementation.

---

## 5. Reproducible final-model reconstruction

Phase 4G may reconstruct the already-frozen 6,144-row final surrogate by reusing the existing:

`run_phase4er_final_refit()`

implementation.

This reconstruction is not model redevelopment.

It must preserve exactly:

- the same 6,144 TRAIN + VALIDATION rows;
- the frozen feature order;
- the frozen target order;
- the frozen density configuration;
- the frozen temperature configuration;
- the frozen random state;
- the frozen target transforms.

TEST targets must not be loaded or accessed.

No Phase 4F evaluation function may be called.

---

## 6. Reproducible loading and inference

Phase 4G must provide a dedicated production loading path.

The loader must validate, before inference:

- manifest schema/version;
- model artifact existence;
- model artifact SHA-256;
- frozen feature names and ordering;
- frozen target identities;
- frozen transforms;
- frozen fit-row count;
- qualified input bounds metadata;
- supported runtime/package compatibility required by the persisted representation.

Inference input must:

- be numeric;
- be finite;
- be two-dimensional for batch inference;
- contain exactly two columns in the frozen order.

The production inference API must return physical-scale predictions:

- density in `m^-3`;
- temperature in `eV`.

Density prediction must therefore apply the inverse `log10` transform exactly as in the existing Phase 4 inference path.

Temperature prediction must preserve the identity transform.

The loader and inference API must not expose TEST targets.

---

## 7. Persistence-equivalence requirement

Before persisted artifacts are accepted, predictions must be compared between:

1. the in-memory frozen final-refit surrogate; and
2. the reloaded persisted surrogate.

The deterministic equivalence probe will use the already-frozen 41 x 41 Phase 4 probe grid:

- 41 equally spaced absorbed-power values from 15 W to 90 W inclusive;
- 41 equally spaced pressure values from 10 mTorr to 60 mTorr inclusive;
- 1,681 total in-domain points.

For both targets Phase 4G must report:

- prediction shape;
- all-finite status;
- exact-array-equality status;
- maximum absolute difference;
- maximum relative difference.

Within the same frozen runtime environment, exact numerical equality is the required acceptance condition.

The equivalence grid is an in-domain persistence/reproducibility test. It is not experimental validation and not OOD validation.

---

## 8. Benchmark purpose

The benchmark measures the implementation-level runtime difference between:

- the canonical reduced-order argon source-simulator evaluation; and
- the frozen surrogate inference path.

It is not an industrial throughput benchmark.

Runtime results are hardware-, software-, operating-system-, and implementation-dependent.

No timing result may alter the benchmark design.

---

## 9. Frozen benchmark operating-point set

The benchmark point set must be independent of TEST target values.

It will use a deterministic regular Cartesian grid containing:

- 8 equally spaced absorbed-power values over 15-90 W inclusive;
- 8 equally spaced pressure values over 10-60 mTorr inclusive.

Total benchmark points:

`64`

Feature ordering must be:

1. absorbed power;
2. pressure.

The benchmark therefore covers the complete numerically qualified source-model envelope without using TEST targets or TEST-derived point selection.

The point set must be generated before any timing begins and must not be changed after timing results are observed.

---

## 10. Source-simulator timing workload

The source benchmark must reuse the canonical Phase 3 simulation path:

`simulate_base_design_point(...)`

with the frozen Phase 3 base-dataset configuration.

For every one of the 64 benchmark operating points, the timed source workload includes:

- construction of the canonical source operating point;
- pressure and pumping-speed setup performed by the canonical simulation path;
- construction of the global-model state;
- execution of `solve_global_model`;
- target extraction;
- the standard source-point validation/calculation path contained in `simulate_base_design_point`.

File loading, configuration loading, benchmark-grid construction, JSON writing, console output, and result serialization are excluded from the timed region.

Every source result must remain numerically qualified. Any source failure must be reported and invalidates the affected benchmark repetition rather than being silently discarded.

---

## 11. Surrogate scalar timing workload

The scalar surrogate workload processes the same ordered 64 operating points one at a time.

For each point the timed workload includes:

- production inference input validation;
- density prediction;
- density inverse target transformation;
- temperature prediction;
- physical-scale result construction.

Model artifact loading is excluded from the inference timing region.

No prediction caching is permitted.

No precomputed predictions are permitted.

---

## 12. Surrogate batch timing workload

The batch surrogate workload processes the same ordered 64 x 2 feature matrix in one inference call.

The timed workload includes:

- production batch-input validation;
- density batch prediction;
- density inverse target transformation;
- temperature batch prediction;
- physical-scale result construction.

Model artifact loading, benchmark-grid construction, JSON writing, and console output are excluded.

---

## 13. Warm-up policy

Warm-up executions are untimed and excluded from reported summaries.

Frozen warm-up counts:

- source simulator: 1 complete 64-point workload;
- surrogate scalar: 5 complete 64-point workloads;
- surrogate batch: 10 complete 64-point workloads.

Warm-up results must not be used to change the protocol.

---

## 14. Timed repetition counts

Frozen timed repetitions:

- source simulator: 3 complete 64-point repetitions;
- surrogate scalar: 200 complete 64-point repetitions;
- surrogate batch: 1,000 complete 64-point repetitions.

Different repetition counts are used only because the runtime scales differ by orders of magnitude.

Every repetition processes the same 64 operating points.

Reported workload runtime is normalized consistently before speed-up calculations.

---

## 15. Timing function

Timing must use:

`time.perf_counter_ns()`

Each repetition must be measured around the complete declared workload.

Raw repetition runtimes must be preserved in the machine-readable result artifact.

---

## 16. Runtime summaries

For each benchmark workload record:

- repetition count;
- benchmark points per repetition;
- raw runtime for every repetition;
- median workload runtime;
- minimum workload runtime;
- maximum workload runtime;
- mean workload runtime;
- standard deviation;
- median runtime per operating point.

The primary runtime comparison is the median.

No fastest-run cherry-picking is permitted.

---

## 17. Speed-up definitions

Primary scalar speed-up:

`median source 64-point runtime / median surrogate scalar 64-point runtime`

Batch speed-up:

`median source 64-point runtime / median surrogate batch 64-point runtime`

The batch result must be explicitly labelled as vectorized batch inference.

Scalar and batch results must not be conflated.

No extrapolation to production fab throughput, wafer throughput, equipment throughput, or customer-response time is permitted.

---

## 18. Benchmark environment capture

The benchmark result must record at minimum:

- operating system/platform;
- machine architecture;
- CPU identity where available;
- Python version;
- NumPy version;
- SciPy version;
- scikit-learn version;
- benchmark timestamp in UTC;
- Git commit;
- repository clean/dirty state;
- benchmark point count;
- benchmark-grid definition;
- warm-up counts;
- timed repetition counts;
- timing function;
- exact included/excluded operations.

The current inspected development environment before benchmark execution is:

- platform: Windows-11-10.0.22631-SP0;
- machine: AMD64;
- CPU identifier: Intel64 Family 6 Model 142 Stepping 9, GenuineIntel;
- Python: 3.12.5;
- NumPy: 2.5.3;
- SciPy: 1.18.1;
- scikit-learn: 1.9.1.

The real benchmark artifact must capture these values dynamically rather than hard-code them from this document.

---

## 19. Benchmark integrity rules

The benchmark implementation must not:

- access TEST targets;
- rerun Phase 4F evaluation;
- select benchmark points based on observed speed;
- change repetition counts after observing timing results;
- change warm-up counts after observing timing results;
- remove slow repetitions because they are inconvenient;
- report only the fastest repetition;
- alter model configuration;
- alter target transformations;
- alter feature ordering;
- change simulator physics;
- parallelize one side solely to improve the reported comparison.

The frozen temperature model remains `n_jobs = 1`.

---

## 20. Machine-readable outputs

Phase 4G will produce auditable machine-readable evidence under the existing repository conventions.

The final Phase 4G outputs must include:

- persisted model binaries under `artifacts/phase4/`;
- a persisted surrogate manifest under `artifacts/phase4/`;
- persistence-equivalence evidence;
- a frozen benchmark specification;
- benchmark results under `results/phase4/`.

Generated production artifacts must not be committed until their automated validation passes.

---

## 21. Automated test requirements

Phase 4G tests must cover at minimum:

- frozen feature ordering;
- correct model identities;
- correct target identities;
- correct target transforms;
- exact frozen hyperparameters;
- 6,144-row final-model reconstruction;
- TEST targets remain inaccessible;
- manifest construction;
- artifact hash verification;
- corrupted-artifact rejection;
- metadata-contract rejection;
- persisted model loading;
- input shape validation;
- non-finite input rejection;
- physical-scale prediction;
- persistence prediction equivalence;
- benchmark-grid determinism;
- benchmark protocol constants;
- timing helper behaviour using controlled test doubles rather than real performance assertions.

Automated tests must never assert that a particular wall-clock speed-up is achieved.

---

## 22. Scientific claim boundary

All Phase 4G artifacts and documentation must preserve the following:

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

The density monotonic constraint represents measured structure of this frozen reduced-order simulator inside the qualified numerical envelope.

It is not a universal ICP law.

The 41 x 41 persistence-equivalence grid and the 8 x 8 speed-benchmark grid are in-domain computational diagnostics only.

They are not experimental validation or OOD validation.

---

## 23. Phase boundary

Phase 4G ends only after:

- the protocol is frozen before timing;
- persistence is implemented and validated;
- provenance is complete;
- persisted/reloaded inference is reproducible;
- persistence equivalence passes;
- the frozen benchmark is executed without protocol modification;
- benchmark evidence is written and validated;
- automated Phase 4G tests pass;
- the resulting artifacts are checkpointed in Git.

Phase 4H remains responsible for final Phase 4 documentation, reproducibility audit, scientific-limitations audit, formal Phase 4 closure, and final Git checkpoint.
