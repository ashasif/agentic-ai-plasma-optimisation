# Phase 3 Dataset and Fault Design

## 1. Dataset architecture

Phase 3 deliberately separates two data products.

### Base steady-state dataset

Purpose:

- surrogate modelling;
- interpolation over the qualified reduced-order input space;
- later constrained optimisation.

Artifact:

`data/synthetic/phase3/base_steady_state.csv`

### Monitoring dataset

Purpose:

- synthetic process monitoring;
- anomaly detection;
- fault detection;
- fault-family classification;
- later agentic diagnostic workflows.

Artifact:

`data/synthetic/phase3/monitoring_episodes.csv`

The monitoring dataset contains observable synthetic measurements and protected simulator/fault ground truth in the same canonical record, while a feature manifest explicitly prevents automatic use of protected columns.

## 2. Base design

The deterministic base dataset uses a two-dimensional scrambled Sobol design.

Dimensions:

- absorbed power: 15?90 W;
- target pressure: 10?60 mTorr.

Fixed flow:

- 20 sccm.

The flow dimension was removed from base surrogate sampling after qualification showed it to be structurally redundant when effective pumping speed is derived from flow and target pressure in the pressure-controlled abstraction.

Flow remains scientifically useful in the monitoring environment because it is then perturbed independently from pumping.

## 3. Base split design

Each split uses its own independently scrambled Sobol sequence.

| Split | Rows | Seed |
|---|---:|---:|
| Train | 4,096 | 20260910 |
| Validation | 2,048 | 20260911 |
| Test | 2,048 | 20260912 |

No failed simulation is silently replaced or resampled.

## 4. Domain status

Base and monitoring rows use the following domain labels:

- `supported`;
- `near_boundary`;
- `ood`.

For the base dataset, a point is near-boundary when at least one normalized sampled dimension lies in the outer 5% of the qualified interval.

Monitoring-domain classification is performed against the true latent absorbed power and true physical pressure.

Sensor noise or sensor bias cannot make an otherwise supported physical state become physics OOD.

## 5. Monitoring episode design

The monitoring dataset contains 64 episodes, each containing 64 ordered quasi-steady rows.

| Split | Episodes | Rows |
|---|---:|---:|
| Train | 32 | 2,048 |
| Validation | 16 | 1,024 |
| Test | 16 | 1,024 |

Split assignment occurs at episode level.

The sequence index is not calibrated physical time.

## 6. Episode allocation

| Episode type | Total |
|---|---:|
| Normal | 16 |
| Power coupling | 12 |
| Flow delivery | 12 |
| Pumping effectiveness | 12 |
| Pressure sensor bias | 12 |

Each fault family contains:

- 4 mild episodes;
- 4 moderate episodes;
- 4 severe episodes;
- 6 step episodes;
- 6 drift episodes.

## 7. Fault severity assumptions

### Power-coupling degradation

Direction:

- negative only.

Magnitudes:

- mild: -5%;
- moderate: -10%;
- severe: -20%.

### Flow-delivery fault

Directions:

- negative;
- positive.

Magnitudes:

- mild: ?3%;
- moderate: ?7%;
- severe: ?15%.

### Pumping-effectiveness fault

Directions:

- negative;
- positive.

Magnitudes:

- mild: ?5%;
- moderate: ?10%;
- severe: ?20%.

### Pressure-sensor bias

Directions:

- negative;
- positive.

Magnitudes:

- mild: ?2%;
- moderate: ?5%;
- severe: ?10%.

These are synthetic scenario magnitudes selected for engineering experimentation.

They are not equipment alarm thresholds, manufacturer tolerances, measured failure distributions, or OIPT specifications.

## 8. Fault timing

Fault onset is selected deterministically between sequence steps 16 and 31 inclusive.

This preserves a pre-fault baseline within each faulty episode.

### Step profile

The full configured fault magnitude is applied at onset.

### Drift profile

Drift uses the cubic smoothstep function:

`g(u) = 3u? - 2u?`

where normalized progress `u` runs from zero at onset to one at the final sequence step.

At the exact onset row:

- `fault_started = true`;
- `fault_progress = 0`;
- `fault_effect_active = false`.

This distinction avoids treating a drift-onset row with zero applied perturbation as an active anomaly.

## 9. Process variability

Normal latent process variability is applied independently at every monitoring row.

| Channel | Sigma | Truncation |
|---|---:|---:|
| Absorbed-power coupling | 1.0% | ?3 sigma |
| Flow delivery | 0.5% | ?3 sigma |
| Pumping effectiveness | 1.0% | ?3 sigma |

Per-row random streams are deterministically derived from:

- top-level seed;
- episode ID;
- step index;
- channel name.

This makes the result invariant to generation order.

## 10. Measurement model

Synthetic measurement channels are:

- absorbed-power estimate;
- gas-flow reading;
- chamber-pressure reading.

Noise assumptions:

| Channel | Sigma | Truncation |
|---|---:|---:|
| Absorbed-power estimate | 1.0% | ?3 sigma |
| Flow | 0.5% | ?3 sigma |
| Pressure | 0.5% | ?3 sigma |

No measured electron-density channel is fabricated.

No measured electron-temperature channel is fabricated.

The simulator values for electron density and temperature remain protected truth.

## 11. Process versus sensor faults

Process faults modify latent simulator inputs.

### Power fault

The absorbed-power input is multiplied by the power-coupling fault factor.

### Flow fault

The true delivered flow changes while pumping is left independently determined.

The pumping speed is not recalculated from the faulted flow.

This preserves the expected pressure response.

### Pumping fault

The true effective pumping speed changes independently while delivered flow remains governed by its own normal variability.

### Pressure-sensor fault

The true physical process is unchanged by the sensor bias.

The bias is applied only after the plasma solution has been calculated.

This enables later models to distinguish physical pressure disturbances from sensor-only pressure discrepancies.

## 12. Monitoring label semantics

### `fault_present`

The episode was configured with a fault.

This remains true even before fault onset.

### `fault_started`

The current row is at or after the configured fault onset.

### `fault_effect_active`

The current fault perturbation has non-zero magnitude.

This is the preferred binary detection target.

### `active_fault_family`

Possible labels:

- `none`;
- `power_coupling`;
- `flow_delivery`;
- `pumping_effectiveness`;
- `pressure_sensor_bias`.

Before a fault has a non-zero effect, the active family is `none`.

## 13. Monitoring feature contract

Approved initial features:

1. `nominal_absorbed_power_W`
2. `target_pressure_mTorr`
3. `nominal_flow_sccm`
4. `measured_absorbed_power_W`
5. `measured_flow_sccm`
6. `measured_pressure_mTorr`

Group metadata:

- `episode_id`;
- `split`.

Ordering metadata:

- `step_index`.

The split and sequence identifiers are not approved predictive features.

## 14. Protected ground truth

Protected fields include:

- fault family;
- fault severity;
- fault direction;
- fault magnitude;
- fault onset;
- fault progression;
- latent fault factors;
- normal process-variability factors;
- true absorbed power;
- true delivered flow;
- true pumping speed;
- true pressure;
- true electron density;
- true electron temperature;
- noise factors;
- pressure-sensor bias;
- solver truth/diagnostics where inappropriate as model inputs.

Downstream ML code must load the feature manifest and must not select all numeric columns automatically.

## 15. Production monitoring QA

Production counts:

- 4,096 rows;
- 64 episodes;
- 3,072 rows belonging to fault-configured episodes;
- 1,024 rows belonging to normal episodes;
- 1,938 rows at or after fault onset;
- 1,914 rows with non-zero active fault effect.

Active-fault row counts:

| Active class | Rows |
|---|---:|
| None | 2,182 |
| Flow delivery | 471 |
| Power coupling | 468 |
| Pressure sensor bias | 483 |
| Pumping effectiveness | 492 |

All 4,096 rows:

- integrated successfully;
- converged;
- passed physical-state checks;
- passed balance checks;
- remained within the supported qualified reduced-order envelope;
- were marked ML eligible.
