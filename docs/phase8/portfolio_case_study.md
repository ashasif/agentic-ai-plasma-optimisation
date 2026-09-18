# Portfolio Case Study - Agentic AI for Reduced-Order ICP Plasma Process Optimisation

## Project at a glance

**Type:** End-to-end scientific machine-learning and decision-support research-engineering project

**Domain:** Synthetic reduced-order argon inductively coupled plasma process analysis

**Core stack:** Python, NumPy, SciPy, scikit-learn, pytest, Git and Git LFS

**Final technical qualification:** 793 / 793 tests passed

**Technical closure:** `a0d57cf8838e294412f97ac2d898ee5aafe968c2`

## The challenge

The project addressed a multi-stage problem: how to move from a synthetic reduced-order plasma model to a reproducible workflow capable of fast surrogate inference, process monitoring, fault diagnosis, constrained operating-point optimisation and human-governed decision support.

A central engineering requirement was to prevent later optimisation and agentic layers from silently changing or bypassing previously validated scientific evidence.

The resulting architecture therefore treats model evaluation, monitoring evidence, optimisation protocols and decision authority as separately frozen and auditable layers.

## What was built

The completed system contains five principal technical layers:

1. **Synthetic monitoring environment** - a reproducible reduced-order argon ICP operating and monitoring foundation.
2. **Surrogate modelling** - persisted models for fast approximation of plasma density and electron temperature.
3. **Monitoring and fault diagnosis** - leakage-safe detection and ambiguity-aware diagnostic inference.
4. **Constrained optimisation** - deterministic operating-point search over a frozen domain using Differential Evolution with a mandatory deterministic grid reference and fallback.
5. **Agentic decision support** - deterministic orchestration of trusted monitoring and optimisation interfaces under fail-closed rules and explicit human approval.

![End-to-end architecture](visuals/end_to_end_architecture.svg)

## Surrogate modelling results

The Phase 4 locked synthetic TEST evaluation achieved:

- density R2 approximately **0.998746**;
- density mean absolute relative error approximately **0.01373**;
- electron-temperature R2 approximately **0.999999994**;
- electron-temperature RMSE approximately **1.06e-05 eV**;
- successful final diagnostic evaluation over **1,681 grid points**.

The inference benchmark recorded approximately **2.94x** scalar surrogate speedup and approximately **149.30x** batched speedup relative to the benchmarked reduced-order source-model workload.

These timing results are implementation-, hardware-, software- and workload-dependent. They are benchmark evidence rather than a universal throughput guarantee.

## Monitoring and fault-diagnosis results

The frozen Phase 5 detector achieved approximately:

- **0.79988 balanced accuracy**;
- **0.80381 macro F1**;
- **0.67819 active-fault recall**;
- **0.87709 positive precision**;
- **0.83131 AUROC**;
- **0.86825 average precision**.

The active-fault recall remained below the frozen 0.75 target and was deliberately preserved rather than tuned after locked evaluation.

All 12 synthetic fault episodes were eventually detected, but the evidence also retained important weaknesses: six fault episodes exhibited pre-active false alarms and all four normal TEST episodes contained at least one false alarm.

The ambiguity-aware diagnoser achieved approximately **0.73632 macro F1**. End-to-end detector-plus-diagnoser macro F1 was approximately **0.68098**, with `pressure_path_anomaly` remaining the weakest final class.

This is an important part of the project: weaker subsystem evidence was preserved rather than hidden behind the much stronger surrogate-model results.

## Constrained optimisation

Phase 6 operates only inside the frozen domain:

- absorbed power: **15-90 W inclusive**;
- pressure: **10-60 mTorr inclusive**.

The deterministic reference grid contains **151 x 101 = 15,251 operating points**.

Differential Evolution is the primary continuous optimiser. The deterministic grid remains mandatory as the reference and fallback mechanism.

The optimisation layer deliberately avoids overstating its results. An accepted solution does not establish global physical optimality, while a no-feasible result means only that no feasible point was found under the frozen search protocol.

Absorbed power is not treated as identical to generator RF power.

## Agentic decision support and governance

![Decision authority and safety flow](visuals/decision_authority_flow.svg)

Phase 7 connects the trusted Phase 5 monitoring runtime and Phase 6 optimisation runtime through deterministic policies.

Key authority rules include:

- trusted-boundary failures fail closed;
- an active fault cannot trigger optimisation without an approved frozen mapping;
- missing mappings request human input rather than being invented;
- operator optimisation requires an explicit operator request;
- recommendations require explicit human approval;
- no autonomous hardware actuation is authorised;
- no autonomous setpoint changes are authorised;
- LLMs have no authoritative scientific, optimisation or actuation authority.

This makes governance part of the actual architecture rather than an external disclaimer.

## Reproducibility and engineering discipline

The repository uses version-controlled protocols, frozen evaluation boundaries, persisted runtime artifacts, deterministic tests and formal phase closure.

The final technical project closed with **793 / 793 tests passed**.

The principal formal closure lineage runs from Phase 3 through Phase 7 before whole-project technical closure. The auditable phase-labelled Git lifecycle begins at Phase 3; no formal Phase 1 or Phase 2 Git closure history is invented.

![Auditable project lifecycle](visuals/project_evidence_lifecycle.svg)

## Technical decisions that mattered

**Locked evaluation instead of retrospective tuning.** Results that missed targets, including Phase 5 active-fault recall, were preserved after locked evaluation.

**Ambiguity-aware diagnosis.** Fault mechanisms that were not sufficiently identifiable using the approved monitoring channels were consolidated rather than presented as reliably distinguishable.

**Reference-grid optimisation.** Continuous optimisation was not trusted in isolation; a deterministic grid remained part of the frozen qualification and fallback contract.

**Fail-closed orchestration.** Phase 7 does not invent scientific evidence, mappings or recovery behaviour when trusted boundaries fail.

**Human authority.** Recommendations remain decision support and require explicit human approval.

## Skills demonstrated

- scientific Python and numerical computing;
- supervised machine learning and model evaluation;
- surrogate modelling and inference benchmarking;
- anomaly detection and fault classification;
- constrained numerical optimisation;
- deterministic runtime design;
- test-driven research engineering;
- model persistence and reproducibility;
- Git-based evidence and release discipline;
- safety and human-authority boundary design;
- technical documentation and scientific communication.

## Limitations

The project is intentionally described within its demonstrated scope:

- synthetic reduced-order argon ICP environment;
- no experimental plasma validation;
- no industrial, fab-scale or wafer-scale validation;
- no validated reactive-chemistry model;
- imperfect monitoring recall and false-alarm behaviour;
- weaker pressure-path diagnosis;
- no global physical optimality guarantee;
- no global physical infeasibility guarantee;
- no autonomous plasma-control deployment.

## Outcome

The project produced a complete and auditable research-engineering pipeline connecting synthetic plasma modelling, high-accuracy surrogate inference, fault-aware monitoring, constrained optimisation and deterministic human-approved decision support.

Its strongest portfolio value is the combination of machine learning, scientific computing, optimisation, software engineering, reproducibility and explicit governance in one end-to-end system.

## Further technical detail

- [Technical project report](technical_project_report.md)
- [Project final summary](../project_final_summary.md)
- [Phase 8C visualisation summary](phase8c_summary.md)

This portfolio case study is a post-closure presentation of frozen project evidence and introduces no new scientific result.
