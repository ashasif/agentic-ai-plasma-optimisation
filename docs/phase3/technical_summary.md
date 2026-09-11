# Phase 3 Technical Summary

## 1. Objective

Phase 3 establishes a reproducible synthetic experimental environment around the Phase 2 reduced-order argon ICP global model.

The phase has two distinct data products:

- a clean deterministic steady-state dataset intended for surrogate-model development; and
- a synthetic monitoring dataset intended for later anomaly detection, fault diagnosis, agentic reasoning, and constrained operating-point workflows.

The distinction between simulator truth and observable synthetic measurements is explicit throughout the design.

## 2. Base operating-space design

The initial design considered absorbed power, target pressure, and gas flow.

A qualification study demonstrated that, under the pressure-controlled abstraction used by the model, varying flow while deriving pumping speed from flow and target pressure produced numerically negligible changes in the steady-state plasma solution.

At fixed absorbed power and pressure, the maximum relative spans observed in the 3 ? 3 ? 3 qualification study were approximately:

| Quantity | Maximum relative flow span |
|---|---:|
| Electron density | 1.77 ? 10^-6 |
| Electron temperature | 6.65 ? 10^-7 |
| Neutral density | 4.63 ? 10^-6 |

The base surrogate design was therefore amended to use two sampled dimensions:

- absorbed power;
- target pressure.

Gas flow was fixed at 20 sccm for the deterministic base dataset and retained as an independent variable for later monitoring/fault scenarios.

## 3. Qualified reduced-order envelope

The numerically qualified sampling envelope is:

| Quantity | Range |
|---|---:|
| Absorbed power | 15?90 W |
| Target pressure | 10?60 mTorr |
| Flow | 20 sccm fixed |
| Gas temperature | 300 K |
| Effective ion-neutral cross-section | 1 ? 10^-18 m? |
| Reactor radius | 0.17 m |
| Reactor length | 0.25 m |

This is a numerically qualified envelope for the project's reduced-order synthetic argon model only.

It is not an experimentally qualified process window and is not an OIPT equipment operating range.

## 4. Envelope qualification

A structured 27-point study used:

- absorbed power: 15, 50, 90 W;
- pressure: 10, 30, 60 mTorr;
- flow: 15, 20, 25 sccm.

All 27 points passed integration, convergence, physical-state, and balance checks.

The expected model trends were recovered:

- increasing pressure increased electron density;
- increasing pressure decreased electron temperature;
- increasing absorbed power increased electron density.

A 256-point scrambled Sobol continuous pilot over the final two-dimensional envelope also passed all numerical qualification checks.

## 5. Deterministic base dataset

The production base dataset contains 8,192 independently simulated steady-state points.

### Sampling

Scrambled Sobol sampling was used independently for each split:

| Split | Rows | Seed |
|---|---:|---:|
| Train | 4,096 | 20260910 |
| Validation | 2,048 | 20260911 |
| Test | 2,048 | 20260912 |

Independent designs were used instead of drawing one design and randomly splitting it.

### Production QA

All 8,192 points passed:

- solver integration;
- convergence;
- physical-state validity;
- quasineutrality;
- particle balance;
- electron-energy balance;
- total-particle identity;
- domain classification;
- ML eligibility.

Observed production ranges were:

| Quantity | Minimum | Maximum |
|---|---:|---:|
| Absorbed power | 15.010951 W | 89.988733 W |
| Target pressure | 10.001137 mTorr | 59.988061 mTorr |
| Electron density | 1.681812 ? 10^16 m^-3 | 2.239906 ? 10^17 m^-3 |
| Electron temperature | 1.547320 eV | 2.074479 eV |

Worst numerical diagnostics:

| Diagnostic | Maximum |
|---|---:|
| Maximum relative rate | 3.869772 ? 10^-7 s^-1 |
| Maximum balance residual | 4.172130 ? 10^-7 |
| Total-particle identity residual | 1.245714 ? 10^-16 |
| Pressure relative error | 1.890456 ? 10^-6 |

The frozen convergence threshold is 1 ? 10^-6 s^-1.

## 6. Base-dataset physical trend audit

Across the complete 8,192-row dataset, Spearman associations were:

| Relationship | Spearman rho |
|---|---:|
| Absorbed power vs electron density | +0.897374 |
| Pressure vs electron density | +0.401622 |
| Pressure vs electron temperature | -1.000000 |
| Absorbed power vs electron temperature | +0.000002 |

Ten pressure bins showed monotonically increasing median electron density and monotonically decreasing median electron temperature.

Ten absorbed-power bins showed monotonically increasing median electron density.

The nearly perfect pressure/electron-temperature relationship and near-zero power/electron-temperature relationship are structural behaviours of this simplified global model and must not be interpreted as universal ICP behaviour.

The maximum relative mismatch between total modelled power losses and absorbed power was approximately 1.96 ? 10^-12.

## 7. Monitoring environment

The monitoring environment contains 64 episodes with 64 ordered quasi-steady steps each:

`64 ? 64 = 4,096 rows`

The steps are ordered sequence indices. They are not calibrated seconds, milliseconds, RF cycles, or experimentally validated physical time increments.

Each monitoring row is independently solved to a quasi-steady state.

### Episode splits

| Split | Episodes | Rows |
|---|---:|---:|
| Train | 32 | 2,048 |
| Validation | 16 | 1,024 |
| Test | 16 | 1,024 |

Entire episodes remain within one split.

No episode-level leakage occurs across train, validation, and test.

## 8. Nominal monitoring envelope

Monitoring recipes use an inner nominal envelope:

| Quantity | Nominal range |
|---|---:|
| Absorbed power | 25?80 W |
| Target pressure | 20?45 mTorr |
| Flow | 20 sccm |

This provides room for synthetic faults and variability while retaining true states inside the broader qualified reduced-order envelope.

The generated monitoring data produced true values of approximately:

| Quantity | Minimum | Maximum |
|---|---:|---:|
| True absorbed power | 19.942811 W | 80.874787 W |
| True pressure | 17.452084 mTorr | 55.485507 mTorr |
| Electron density | 3.677252 ? 10^16 m^-3 | 1.754819 ? 10^17 m^-3 |
| Electron temperature | 1.564868 eV | 1.878138 eV |

All 4,096 rows remained classified as supported within the Phase 3B reduced-order envelope.

## 9. Monitoring variability and measurements

Normal synthetic process variability:

| Latent quantity | Relative sigma |
|---|---:|
| Absorbed-power coupling | 1.0% |
| Flow delivery | 0.5% |
| Pumping effectiveness | 1.0% |

Synthetic measurement noise:

| Measurement | Relative sigma |
|---|---:|
| Absorbed-power estimate | 1.0% |
| Flow | 0.5% |
| Pressure | 0.5% |

All Gaussian variability/noise is truncated at ?3 sigma.

These magnitudes are engineering scenario assumptions, not OIPT specifications or experimentally calibrated uncertainty models.

## 10. Fault environment

The monitoring environment contains:

- 16 normal episodes;
- 12 power-coupling fault episodes;
- 12 flow-delivery fault episodes;
- 12 pumping-effectiveness fault episodes;
- 12 pressure-sensor-bias episodes.

Across the 48 fault episodes:

- mild: 16;
- moderate: 16;
- severe: 16;
- step profile: 24;
- smooth drift: 24.

No compound faults are included in Phase 3.

## 11. Scientific fault-signature results

Independent post-generation auditing confirmed that each synthetic fault family creates the intended physical or observational signature.

Median pre-fault-to-final-window changes were:

| Fault | Direction | Principal injected/observed signature |
|---|---|---|
| Power coupling | Negative | True power -10.37%, electron density -10.15% |
| Flow delivery | Negative | True flow -6.90%, true pressure -6.64% |
| Flow delivery | Positive | True flow +10.63%, true pressure +11.38% |
| Pumping effectiveness | Negative | True pumping -11.79%, true pressure +14.05% |
| Pumping effectiveness | Positive | True pumping +10.22%, true pressure -9.32% |
| Pressure sensor bias | Negative | True pressure ~+0.31%, measured pressure -6.68% |
| Pressure sensor bias | Positive | True pressure ~-0.23%, measured pressure +4.43% |

The sensor-fault cases are particularly important: measured pressure changes strongly while the underlying physical pressure and plasma state remain approximately unchanged except for normal synthetic variability.

## 12. Cold-start strategy

Every monitoring step is solved independently.

The initial neutral density is based on the pressure implied by the row's true flow-to-pumping ratio rather than inherited from the previous monitoring row.

This physics-informed cold start was introduced after a diagnostic showed that starting faulted rows from the commanded target pressure caused slow neutral-inventory relaxation and false non-convergence within the fixed 20 s solver horizon.

With the corrected independent Q/S initialization, the representative diagnostic cases converged with maximum relative rates around 10^-9 s^-1 while preserving:

- the 20 s solver horizon;
- the 1 ? 10^-6 s^-1 convergence threshold;
- independent row solves;
- the fault design.

No convergence tolerance was weakened.

## 13. Feature contracts

### Base surrogate features

The base feature contract uses:

- `nominal_absorbed_power_W`;
- `target_pressure_mTorr`.

Primary targets:

- `true_electron_density_m3`;
- `true_electron_temperature_eV`.

### Monitoring features

Only the following are approved as initial monitoring ML features:

- `nominal_absorbed_power_W`;
- `target_pressure_mTorr`;
- `nominal_flow_sccm`;
- `measured_absorbed_power_W`;
- `measured_flow_sccm`;
- `measured_pressure_mTorr`.

Binary target:

- `fault_effect_active`.

Multiclass target:

- `active_fault_family`.

True physical states, latent variability, injected fault parameters, measurement-noise realizations, and diagnostic truth fields are protected from model inputs.

## 14. Reproducibility

Canonical CSV artifacts use deterministic dataclass-field ordering, LF line endings, stable Boolean formatting, and stable floating-point formatting.

The base dataset SHA-256 is:

`dc8d3a24f84b3eca9ce930a88e969e51892d69430ce5840cc546523e9f9dd423`

The monitoring dataset SHA-256 is:

`ee20fe9e6ffac4875911cba23247abeeeba236954349c1696cc1cb1ee59c7a3a`

The repository `.gitattributes` forces canonical Phase 3 CSV and JSON artifacts to use LF line endings so their byte-level hashes are portable across Windows Git checkouts.

## 15. Final QA

At the end of Phase 3:

- 193 automated tests pass;
- both production dataset hashes match their manifests;
- all required artifacts exist;
- all canonical artifacts use LF;
- all base rows are valid and ML eligible;
- all monitoring rows are valid and ML eligible;
- no episode crosses dataset splits;
- the monitoring feature contract contains no protected-ground-truth leakage;
- all independently audited fault signatures pass;
- the working tree was clean immediately before documentation generation.
