# Phase 3 Closure Checklist

## Phase 3A ? Experimental design

- [x] Separate deterministic simulator data from monitoring/fault data.
- [x] Define pressure-controlled reactor abstraction.
- [x] Investigate flow redundancy.
- [x] Amend base surrogate design to absorbed power plus pressure.
- [x] Fix deterministic base flow at 20 sccm.
- [x] Retain flow variation for monitoring/fault experiments.
- [x] Freeze synthetic variability/noise assumptions.
- [x] Freeze fault families and severities.
- [x] Exclude unsupported substrate/bias fault family.
- [x] Freeze episode-level split policy.
- [x] Freeze leakage-safe feature policy.

## Phase 3B ? Operating-envelope qualification

- [x] Complete 27-point structured qualification.
- [x] Verify integration and convergence.
- [x] Verify particle and energy balances.
- [x] Verify physical-state bounds.
- [x] Confirm pressure-up / electron-density-up trend.
- [x] Confirm pressure-up / electron-temperature-down trend.
- [x] Confirm power-up / electron-density-up trend.
- [x] Complete 256-point continuous Sobol pilot.
- [x] Freeze 15?90 W and 10?60 mTorr reduced-order sampling envelope.
- [x] Document envelope as synthetic/model-specific rather than industrial.

## Phase 3C ? Base dataset architecture

- [x] Freeze independent train/validation/test Sobol designs.
- [x] Freeze split seeds.
- [x] Define canonical simulation IDs.
- [x] Define 42-column base schema.
- [x] Preserve integration/convergence failures.
- [x] Define supported / near-boundary / OOD status.
- [x] Define model-validity status.
- [x] Freeze base feature manifest.
- [x] Prevent numeric-column auto-selection.
- [x] Implement canonical CSV serialization.
- [x] Implement semantic configuration hash.
- [x] Implement physics-source hash.
- [x] Implement deterministic dataset SHA-256.

## Phase 3D ? Base dataset production

- [x] Generate 8,192 production rows.
- [x] Verify 8,192 unique IDs.
- [x] Verify 8,192 unique coordinates.
- [x] Verify split counts 4,096 / 2,048 / 2,048.
- [x] Verify all rows integration-successful.
- [x] Verify all rows converged.
- [x] Verify all rows physically valid.
- [x] Verify quasineutrality.
- [x] Verify particle and electron-energy balances.
- [x] Verify all rows ML eligible.
- [x] Audit global physical/statistical trends.
- [x] Verify canonical LF serialization.
- [x] Verify dataset SHA-256.
- [x] Freeze production dataset in Git.

Base dataset hash:

`dc8d3a24f84b3eca9ce930a88e969e51892d69430ce5840cc546523e9f9dd423`

Base artifact commit:

`558ef3f`

## Phase 3E ? Monitoring and fault environment

- [x] Freeze 64 episodes ? 64 ordered quasi-steady steps.
- [x] Freeze episode-level train/validation/test split.
- [x] Generate 64 unique nominal recipes.
- [x] Implement deterministic per-row random streams.
- [x] Implement normal process variability.
- [x] Implement measurement noise.
- [x] Implement step-fault profile.
- [x] Implement cubic smooth-drift profile.
- [x] Implement power-coupling degradation.
- [x] Implement flow-delivery faults.
- [x] Implement pumping-effectiveness faults.
- [x] Implement pressure-sensor bias.
- [x] Keep sensor faults out of latent plasma physics.
- [x] Preserve independent flow/pumping semantics.
- [x] Implement physics-informed independent cold starts.
- [x] Keep 20 s solver horizon unchanged.
- [x] Keep 1 ? 10^-6 s^-1 convergence threshold unchanged.
- [x] Freeze 76-column monitoring schema.
- [x] Freeze leakage-safe monitoring feature manifest.
- [x] Generate 4,096 production rows.
- [x] Verify all 4,096 rows integration-successful.
- [x] Verify all 4,096 rows converged.
- [x] Verify all 4,096 rows physically valid.
- [x] Verify all 4,096 rows balance-valid.
- [x] Verify all 4,096 rows supported.
- [x] Verify all 4,096 rows ML eligible.
- [x] Verify fault-label mathematics.
- [x] Verify process-fault injection equations.
- [x] Verify measurement equations.
- [x] Verify process/noise truncation limits.
- [x] Verify no episode split leakage.
- [x] Verify no protected-ground-truth feature leakage.
- [x] Verify independent scientific fault signatures.
- [x] Verify canonical LF serialization.
- [x] Verify dataset SHA-256.
- [x] Freeze monitoring artifacts in Git.

Monitoring dataset hash:

`ee20fe9e6ffac4875911cba23247abeeeba236954349c1696cc1cb1ee59c7a3a`

Monitoring artifact commit:

`64b3adc`

## Phase 3F ? Final QA and documentation

- [x] Full automated regression suite passes: 193 tests.
- [x] Cross-artifact QA passes.
- [x] Both dataset hashes match frozen manifests.
- [x] Both feature contracts are present.
- [x] Monitoring episodes remain split-safe.
- [x] Scientific-scope caveats retained.
- [x] Technical summary written.
- [x] Dataset/fault-design documentation written.
- [x] Scientific limitations documentation written.
- [x] Final documentation QA.
- [x] Final Phase 3 documentation Git checkpoint.
- [x] Confirm clean working tree after final commit.
- [x] Formally declare Phase 3 closed.

## Required claims discipline for later phases

The following statements remain mandatory:

- the model is zero-dimensional and reduced order;
- the gas is argon;
- the bulk model is electropositive and quasineutral;
- the EEDF treatment is simplified;
- detailed metastable and reactive chemistry is absent;
- detailed sheath dynamics are absent;
- spatial gas-flow/temperature fields are absent;
- wafer uniformity is not predicted;
- etch rate, selectivity, and profile are not predicted;
- absorbed power is not generator RF power;
- process variability, measurement noise, and fault severities are synthetic assumptions;
- the sampled envelope is not an OIPT equipment operating range;
- no experimental or industrial validation has been performed.
