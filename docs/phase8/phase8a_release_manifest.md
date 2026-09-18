# Phase 8A — Release Packaging Manifest

## 1. Purpose

This document records the packaging boundary for the post-closure release of:

**Agentic AI for Reduced-Order ICP Plasma Process Monitoring, Fault Diagnosis and Constrained Operating-Point Optimisation**

Phase 8A is dissemination and release packaging only.

It does not reopen the scientific/software project.

---

## 2. Immutable technical baseline

Formal project closure commit:

`a0d57cf8838e294412f97ac2d898ee5aafe968c2`

Subject:

`docs: formally close project`

This commit is the immutable technical baseline for Phase 8.

Phase 8 commits must descend from this baseline and must be identifiable as packaging, documentation, dissemination, or professional-presentation work.

---

## 3. Qualification baseline

The formally closed repository qualification is:

- 793 / 793 tests passed;
- working tree clean;
- final Phase 3–7 technical lineage intact;
- frozen Phase 5, Phase 6 and Phase 7 identities preserved.

The Phase 8A pre-release reproducibility audit independently reproduced:

- Python: 3.12.5;
- package: `plasma-ai` 0.1.0;
- complete regression: 793 passed;
- pytest exit code: 0;
- repository unchanged after execution.

---

## 4. Historical evidence boundary

The auditable phase-labelled Git history begins with Phase 3.

Formal closure lineage:

| Stage | Closure commit |
| --- | --- |
| Phase 3 | `247374cf93d4335c81ef47956941b6918ea6c5e3` |
| Phase 4 | `853024fbc76f3fc046cab018bf57002bba80263c` |
| Phase 5 | `7f8f9b32c8551f422283d996cd358e729146181a` |
| Phase 6 | `08aa55e22b5c8d8cf7c397797481d6e94651dbc5` |
| Phase 7 | `a6046186edf33985475cce7dc665e537f9132965` |
| Project | `a0d57cf8838e294412f97ac2d898ee5aafe968c2` |

No Git commits were found explicitly labelled Phase 1 or Phase 2.

Phase 8 public material must not invent formal Phase 1 or Phase 2 implementation, qualification, or closure claims.

---

## 5. Release identity

Current package metadata:

- package name: `plasma-ai`;
- package version: `0.1.0`;
- supported Python declaration: `>=3.11`.
- software licence: MIT License.

Candidate release tag:

`v0.1.0`

No tag is authorised until Phase 8A release contents have been reviewed and committed.

---

## 6. Artifact distribution policy

The repository contains frozen scientific/model artifacts.

Phase 4 persisted surrogate artifacts are managed through Git LFS:

- `artifacts/phase4/density_model.pkl`
  - LFS SHA-256: `2706589929cdbebabda82cc9091ce110d673adc1c380bb49de9590f89da0b6a8`

- `artifacts/phase4/temperature_model.pkl`
  - LFS SHA-256: `891c5f99082ae27f8dd08d610d528460ecae72e4d019c0bc47ba0f993420c7e5`

A complete public clone therefore requires Git LFS.

The Phase 5 detector and diagnoser are direct Git objects:

- `artifacts/phase5/fault_detector.pkl`
- `artifacts/phase5/fault_diagnoser.pkl`

Phase 8 does not rewrite Git history merely to migrate those frozen artifacts to LFS.

Generated/local development content such as `.venv`, `.pytest_cache`, Python bytecode caches, local egg-info metadata, `.vscode`, `.env`, build output and distribution output remains excluded through `.gitignore`.

---

## 7. Scientific claim boundary

The release concerns a synthetic reduced-order argon ICP research-engineering system.

It does not establish:

- experimental validation;
- industrial validation;
- fab-scale or wafer-scale validation;
- semiconductor-production validation;
- reactive-chemistry validity;
- global physical optimality;
- global physical infeasibility;
- autonomous hardware actuation;
- autonomous setpoint control.

Absorbed power is model absorbed power and is not asserted to be identical to generator RF power.

The frozen Phase 6 operating domain is:

- absorbed power: 15–90 W inclusive;
- pressure: 10–60 mTorr inclusive.

No-feasible optimisation results mean only that no feasible point was found under the frozen search protocol.

Phase 7 remains deterministic decision support.

Human approval is required for recommendations.

LLMs have no authority in the authoritative scientific decision path.

---

## 8. Frozen technical content

Phase 8A must not rewrite for presentation:

- Phase 3 synthetic scientific evidence;
- Phase 4 surrogate qualification evidence;
- persisted Phase 4 models;
- Phase 5 monitoring/fault-diagnosis evidence;
- persisted Phase 5 models;
- Phase 6 optimisation evidence;
- Phase 6 production runtime behaviour;
- Phase 7 deterministic decision policy;
- Phase 7 trusted adapters;
- Phase 7 orchestration behaviour;
- final project closure evidence.

---

## 9. Phase 8A packaging scope

Permitted packaging work includes:

- release metadata;
- citation metadata;
- release notes;
- licence metadata after an explicit licence decision;
- GitHub repository metadata;
- installation/reproducibility presentation;
- public documentation links;
- audited release tagging;
- audited GitHub release creation.

Deferred work:

- comprehensive public README — Phase 8B;
- architecture diagrams — Phase 8C;
- technical/dissertation-style report — Phase 8D;
- professional case study — Phase 8E;
- CV, LinkedIn and application packaging — Phase 8F.

---

## 10. Outstanding release decisions

Before public release:

1. MIT License selected and recorded in `LICENSE` and package/citation metadata;
2. create the Phase 8A packaging commit(s);
3. public GitHub repository created and `origin` configured: `https://github.com/ashasif/agentic-ai-plasma-optimisation`;
4. empty public remote audited before first push;
5. repository URLs recorded in package and citation metadata;
6. complete Phase 8B README work;
7. audit final release contents;
8. create the release tag only after the audited content is approved;
9. push Git/LFS objects;
10. verify the public clone/reproducibility path;
11. create the GitHub release.

No item in this document authorises modification of frozen scientific evidence.
