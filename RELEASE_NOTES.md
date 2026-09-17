# Release Notes

## v0.1.0 — Formal Technical Closure Release Candidate

This release candidate packages the formally closed technical baseline of:

**Agentic AI for Reduced-Order ICP Plasma Process Monitoring, Fault Diagnosis and Constrained Operating-Point Optimisation**

The immutable technical closure baseline is:

`a0d57cf8838e294412f97ac2d898ee5aafe968c2`

Commit subject:

`docs: formally close project`

The package version declared by `pyproject.toml` is `0.1.0`. The natural release-tag candidate is therefore `v0.1.0`.

No release tag has yet been created.

## Technical qualification

The final whole-repository qualification at technical closure is:

- 793 tests passed;
- 0 test failures;
- final technical closure commit recorded after qualification;
- critical frozen Phase 5, Phase 6 and Phase 7 identities preserved.

Phase 8 packaging work is downstream of the formally closed technical baseline and does not reopen the scientific/software qualification.

## Version-controlled lifecycle

The auditable phase-labelled Git history begins with Phase 3.

Formal high-level closure lineage:

- Phase 3: `247374cf93d4335c81ef47956941b6918ea6c5e3`
- Phase 4: `853024fbc76f3fc046cab018bf57002bba80263c`
- Phase 5: `7f8f9b32c8551f422283d996cd358e729146181a`
- Phase 6: `08aa55e22b5c8d8cf7c397797481d6e94651dbc5`
- Phase 7: `a6046186edf33985475cce7dc665e537f9132965`
- Final project closure: `a0d57cf8838e294412f97ac2d898ee5aafe968c2`

The final audit found no Git commits explicitly labelled Phase 1 or Phase 2. This release therefore does not claim that Phase 1 or Phase 2 were separately implemented, qualified, or formally closed as version-controlled phases.

## Technical scope

The repository implements a reproducible research-engineering workflow for a **synthetic reduced-order argon inductively coupled plasma (ICP) system**.

The version-controlled Phase 3–7 lifecycle covers:

- synthetic operating-envelope and monitoring-data generation;
- reduced-order plasma simulation;
- surrogate modelling;
- physics-aware surrogate qualification;
- monitoring and fault diagnosis;
- constrained operating-point optimisation;
- deterministic optimisation fallback logic;
- trusted monitoring and optimisation runtime boundaries;
- deterministic agentic decision support;
- immutable decision recording and replay;
- explicit human-approval boundaries.

## Scientific and operational boundaries

This release must not be interpreted as evidence of:

- experimental plasma validation;
- industrial process validation;
- semiconductor-production validation;
- fab-scale validation;
- wafer-scale validation;
- reactive-chemistry validity;
- global physical optimality;
- global physical infeasibility;
- autonomous hardware actuation;
- autonomous setpoint control.

The project uses synthetic and model-based evidence.

`absorbed_power_W` represents model absorbed power and must not be interpreted as identical to generator RF power.

The frozen Phase 6 operating domain is:

- absorbed power: 15–90 W inclusive;
- pressure: 10–60 mTorr inclusive.

A Phase 6 no-feasible result means only that no feasible point was found under the frozen search protocol. It does not establish global physical infeasibility.

Phase 7 is deterministic decision support rather than autonomous control. Recommendations require human approval.

Large language models have no authority in the authoritative scientific decision path.

## Reproducibility

The package declares Python `>=3.11`.

The final Phase 8A pre-release audit reproduced the complete 793-test qualification using Python 3.12.5.

The repository uses Git LFS for the frozen Phase 4 persisted surrogate artifacts. A complete public clone therefore requires Git LFS support.

The Phase 4 Git LFS objects are:

- `artifacts/phase4/density_model.pkl`
- `artifacts/phase4/temperature_model.pkl`

The Phase 5 persisted detector and diagnoser remain direct Git objects. Their history is not rewritten during Phase 8 packaging.

## Release packaging status

At creation of this document:

- technical project: formally closed;
- candidate package version: `0.1.0`;
- candidate tag: `v0.1.0`;
- Git tag: not yet created;
- GitHub remote: not yet configured;
- GitHub release: not yet created;
- public README: deferred to Phase 8B;
- architecture visualisation: deferred to Phase 8C;
- licence selection: pending explicit author decision.

Phase 8 packaging commits may improve presentation and release metadata, but they must remain clearly downstream of the immutable formal technical closure baseline.
