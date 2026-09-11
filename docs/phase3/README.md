# Phase 3 ? Synthetic Experimental Environment and Dataset Design

## Status

Phase 3 is formally complete and closed.

Implementation, production dataset generation, numerical QA, scientific fault-signature QA, cross-artifact validation, documentation QA, and final regression verification have all passed.

## Purpose

Phase 3 converts the reduced-order argon ICP model developed in Phase 2 into two reproducible synthetic-data environments:

1. a deterministic steady-state surrogate dataset for later modelling and optimisation; and
2. an ordered quasi-steady monitoring dataset containing controlled synthetic process variability, measurement noise, drift, and single-fault scenarios.

The work is intended as a research-engineering and portfolio demonstration of reproducible simulation-data design, monitoring-data generation, fault injection, validation, and leakage-safe ML preparation.

It is not an experimental plasma dataset, CFD model, industrial digital twin, semiconductor etch model, or Oxford Instruments Plasma Technology operating envelope.

## Phase 3 deliverables

| Deliverable | Status |
|---|---|
| Experimental-design decisions frozen | Complete |
| Reduced-order sampling envelope numerically qualified | Complete |
| Deterministic base surrogate dataset | Complete |
| Monitoring episode design | Complete |
| Synthetic process variability | Complete |
| Synthetic measurement noise | Complete |
| Step and smooth-drift fault profiles | Complete |
| Four single-fault families | Complete |
| Leakage-safe feature contracts | Complete |
| Canonical CSV serialization and SHA-256 provenance | Complete |
| Independent numerical QA | Complete |
| Independent scientific fault-signature QA | Complete |
| Full repository regression suite | 193 tests passing |
| Final documentation Git checkpoint | Complete |

## Frozen production artifacts

### Deterministic base dataset

- Rows: 8,192
- Train: 4,096
- Validation: 2,048
- Test: 2,048
- Qualification-valid rows: 8,192
- ML-eligible rows: 8,192
- SHA-256:

`dc8d3a24f84b3eca9ce930a88e969e51892d69430ce5840cc546523e9f9dd423`

### Monitoring dataset

- Episodes: 64
- Ordered quasi-steady steps per episode: 64
- Rows: 4,096
- Train: 2,048 rows / 32 episodes
- Validation: 1,024 rows / 16 episodes
- Test: 1,024 rows / 16 episodes
- Qualification-valid rows: 4,096
- ML-eligible rows: 4,096
- SHA-256:

`ee20fe9e6ffac4875911cba23247abeeeba236954349c1696cc1cb1ee59c7a3a`

## Documentation

- [Technical summary](technical_summary.md)
- [Dataset and fault design](dataset_and_fault_design.md)
- [Scientific scope and limitations](scientific_scope_and_limitations.md)
- [Closure checklist](closure_checklist.md)

## Key repository checkpoints

| Commit | Purpose |
|---|---|
| `f4f1bbb` | Qualify Phase 3 synthetic operating envelope |
| `a25ccb8` | Add reproducible Phase 3 base dataset pipeline |
| `4116305` | Add Phase 3 base dataset artifact generator |
| `751ef1c` | Normalize base-dataset validity flags |
| `558ef3f` | Freeze validated Phase 3 base synthetic dataset |
| `c1e828a` | Add Phase 3 synthetic monitoring environment |
| `64b3adc` | Freeze validated Phase 3 monitoring dataset |
