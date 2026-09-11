# Phase 3 Scientific Scope and Limitations

## 1. Intended scope

Phase 3 is a synthetic research-engineering environment built on the project's Phase 2 zero-dimensional reduced-order argon ICP global model.

It demonstrates:

- controlled experimental-design reasoning;
- synthetic simulation-data generation;
- reproducible sampling;
- numerical validation;
- synthetic monitoring design;
- process-versus-sensor fault injection;
- deterministic stochastic simulation;
- provenance management;
- leakage-safe ML dataset preparation.

It does not demonstrate experimental validation of an industrial plasma reactor.

## 2. Spatial dimensionality

The underlying model is zero-dimensional.

It represents volume-averaged plasma quantities rather than spatial fields.

It therefore does not directly model:

- radial plasma profiles;
- axial plasma profiles;
- wafer-scale uniformity;
- local sheath structure;
- local gas velocity;
- local temperature fields;
- local species distributions;
- chamber geometry effects resolved by CFD.

Phase 3 must not be described as CFD, PIC, or a spatially resolved digital twin.

## 3. Chemistry

The current model is an argon, electropositive, reduced-order global model.

It does not contain a detailed semiconductor plasma-chemistry mechanism.

Specifically, Phase 3 does not model:

- fluorocarbon chemistry;
- chlorine chemistry;
- oxygen-containing reactive chemistry;
- molecular dissociation networks;
- metastable-state kinetics;
- negative-ion chemistry;
- wafer-surface reaction mechanisms;
- deposition chemistry;
- etch-product chemistry.

## 4. Sheath and surface physics

Charged-particle wall losses and wall energy losses are represented through reduced-order expressions.

The model does not provide detailed:

- RF sheath dynamics;
- ion-energy distribution functions;
- ion-angle distributions;
- substrate-bias waveform response;
- feature-scale surface evolution.

A substrate or bias-power fault family was therefore deliberately excluded from Phase 3.

## 5. Electron kinetics

Electron kinetics use simplified reduced-order rate models consistent with the Phase 2 assumptions.

The project does not resolve a full non-Maxwellian electron-energy distribution function.

The electron-temperature response is therefore a model state, not a direct experimental observable.

## 6. Power interpretation

`absorbed_power_W` is model absorbed power.

It is not equivalent to generator RF power.

The synthetic monitoring measurement named measured absorbed power should be interpreted as a synthetic absorbed-power estimate.

No claim is made that a real commercial tool directly measures absorbed plasma power in this way.

## 7. Flow and pumping interpretation

The deterministic base dataset uses a pressure-controlled abstraction in which pumping speed is derived consistently from nominal flow and target pressure.

This makes flow structurally redundant at fixed target pressure within this simplified steady-state formulation.

The monitoring environment deliberately breaks that dependency when constructing faults so that delivered flow and pumping effectiveness can vary independently.

This is a synthetic fault experiment, not a detailed pressure-control-system model.

## 8. Time interpretation

Monitoring rows are ordered quasi-steady sequence steps.

They are not calibrated units of physical time.

The monitoring environment does not simulate:

- controller response times;
- valve dynamics;
- RF matching-network dynamics;
- hardware latency;
- chamber thermal transients;
- actual fault propagation times.

A drift profile specifies progression through sequence position, not a measured drift rate in seconds.

## 9. Cold-start interpretation

Each monitoring state is independently solved.

The solver uses a physics-informed neutral initial condition based on the true row-specific flow-to-pumping ratio.

This improves convergence without introducing dependence on the preceding row.

The episode ordering therefore controls synthetic fault progression and analysis structure, not ODE state inheritance.

## 10. Synthetic variability and noise

The normal process-variability and measurement-noise magnitudes used in Phase 3 are scenario assumptions.

They are not derived from:

- OIPT equipment specifications;
- measured chamber logs;
- metrology studies;
- repeatability studies;
- sensor calibration certificates;
- manufacturing statistical-process-control data.

They should be described explicitly as synthetic engineering assumptions.

## 11. Synthetic fault severities

Fault magnitudes were selected to create mild, moderate, and severe synthetic scenarios.

They are not:

- measured failure distributions;
- equipment alarm limits;
- preventative-maintenance thresholds;
- process-control limits;
- safety thresholds;
- industrial acceptance criteria.

## 12. Single-fault restriction

Phase 3 contains one fault family per faulty episode.

Compound or interacting faults are excluded.

Real equipment can exhibit coupled disturbances, correlated degradation, actuator/sensor interactions, and multiple simultaneous root causes.

Later work may study compound scenarios, but Phase 3 results cannot establish performance for them.

## 13. Model structural relationships

The Phase 3 base dataset showed an almost perfectly monotonic relationship between pressure and electron temperature:

Spearman rho approximately -1.0.

It also showed almost no dependence of electron temperature on absorbed power over the sampled design:

Spearman rho approximately 0.

These are structural features of the present reduced-order formulation.

They must not be generalized to all ICP systems or interpreted as experimentally established plasma laws over industrial process windows.

## 14. Etch and deposition outputs

The current model predicts plasma-state quantities, not semiconductor process outcomes.

Phase 3 does not predict:

- etch rate;
- deposition rate;
- selectivity;
- critical dimension;
- sidewall profile;
- surface roughness;
- wafer non-uniformity;
- defectivity.

No fabricated process-response target was introduced in Phase 3.

If a later phase introduces an engineering objective/index, it must be explicitly identified as synthetic unless supported by external physical or experimental evidence.

## 15. Experimental and industrial validation

The Phase 3 datasets contain no experimental measurements.

They have not been calibrated against a specific OIPT chamber.

They do not establish:

- industrial predictive accuracy;
- process-window qualification;
- chamber matching;
- recipe transferability;
- equipment health thresholds;
- production readiness.

The project should therefore use phrases such as:

- synthetic environment;
- reduced-order model;
- numerically qualified model envelope;
- simulated monitoring data;
- synthetic fault scenarios.

It should avoid phrases such as:

- experimentally validated digital twin;
- industrial plasma model;
- OIPT process model;
- production-ready optimiser;
- experimentally proven fault detector.

## 16. Relationship to the target KTP role

This project is relevant to the target role because it demonstrates the workflow around physics-aware simulation, reproducible engineering data, surrogate-ready datasets, monitoring data, fault diagnosis preparation, constrained optimisation preparation, and agentic-system foundations.

However, it is intentionally narrower than the KTP role's intended industrial modelling environment.

It should be presented as evidence of transferable research-engineering capability rather than as a reproduction of the Oxford Instruments/University of Exeter system.
