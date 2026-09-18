# Agentic AI for Reduced-Order ICP Plasma Process Monitoring, Fault Diagnosis and Constrained Operating-Point Optimisation

A research-engineering project that combines a **synthetic reduced-order argon inductively coupled plasma (ICP) environment** with physics-aware surrogate modelling, process monitoring and fault diagnosis, constrained operating-point optimisation, and deterministic human-approved agentic decision support.

## 1. Project identity

The project demonstrates an end-to-end engineering workflow around a reduced-order plasma model rather than an industrial plasma reactor or production digital twin.

The implemented system connects four technical capabilities:

- reproducible synthetic plasma-process data generation;
- persisted surrogate modelling of electron density and electron temperature;
- trusted monitoring, fault detection and fault diagnosis;
- constrained operating-point optimisation and deterministic decision support.

The auditable phase-labelled Git lifecycle represented in this repository begins at **Phase 3** and continues through the formal closure of **Phase 7**.

## 2. Project status and qualification

| Item | Status |
|---|---|
| Technical implementation | Formally closed |
| Final whole-project regression | **793 tests passed** |
| Package | `plasma-ai` |
| Package version | `0.1.0` |
| Python | `>=3.11` |
| Licence | MIT |
| Repository | Public |
| Technical closure commit | `a0d57cf8838e294412f97ac2d898ee5aafe968c2` |

Post-closure work is limited to packaging, documentation, visualisation and professional presentation. It does not silently reopen frozen scientific evidence, models, protocols, optimisation behaviour or decision logic.

See [`docs/project_final_summary.md`](docs/project_final_summary.md) for the formal whole-project closure record.

## 3. What the system does

The project builds a controlled decision-support stack over a synthetic reduced-order argon ICP environment.

At a high level, it:

1. constructs reproducible synthetic plasma-process and monitoring data;
2. trains and freezes surrogate models for electron density and electron temperature;
3. performs process monitoring, binary fault detection and fault-family diagnosis;
4. searches a constrained operating domain using deterministic and continuous optimisation methods;
5. exposes trusted Phase 5 and Phase 6 runtime boundaries;
6. applies deterministic Phase 7 decision policy and orchestration;
7. produces recommendations that remain **pending human approval**.

The authoritative scientific path does not delegate prediction, optimisation authority or hardware-actuation authority to an LLM.

## 4. End-to-end technical workflow

The implemented workflow is:

**Synthetic reduced-order ICP environment**<br>
→ **surrogate modelling and locked TEST evaluation**<br>
→ **monitoring and fault diagnosis**<br>
→ **constrained operating-point optimisation**<br>
→ **trusted fail-closed runtime boundaries**<br>
→ **deterministic agentic decision support**<br>
→ **human approval or escalation**

Each major capability was developed under an explicit evidence and governance boundary so that later phases could consume frozen upstream interfaces without silently retraining, retuning or rewriting them.

## 5. Auditable implementation lifecycle

| Phase | Main purpose | Formal closure |
|---|---|---|
| Phase 3 | Synthetic experimental environment and monitoring dataset | `247374c` |
| Phase 4 | Physics-aware surrogate modelling, validation, persistence and inference benchmarking | `853024f` |
| Phase 5 | Monitoring, fault detection, fault diagnosis and trusted inference | `7f8f9b3` |
| Phase 6 | Constrained operating-point optimisation and production runtime | `08aa55e` |
| Phase 7 | Trusted deterministic agentic decision support and orchestration | `a604618` |
| Whole project | Final technical audit and formal closure | `a0d57cf` |

The repository deliberately preserves the distinction between technical closure and later presentation/release packaging.

## 6. Key quantitative evidence

The numbers below are frozen project evidence, not claims of experimental or industrial performance.

| Capability | Frozen evidence |
|---|---|
| Phase 4 density surrogate | TEST R² = **0.998746**; mean absolute relative error = **0.01373** |
| Phase 4 temperature surrogate | TEST R² = **0.999999994**; RMSE = **1.06 × 10⁻⁵ eV** |
| Phase 4 batch inference benchmark | **1.403 ms/point**, approximately **149.3×** faster than the reduced-order source workload in the recorded benchmark |
| Phase 5 fault detector | balanced accuracy = **0.7999**; macro F1 = **0.8038**; AUROC = **0.8313** |
| Phase 5 fault diagnosis | macro F1 = **0.7363**; balanced accuracy = **0.7532** |
| Phase 5 sequence evidence | all **12** synthetic fault episodes were eventually detected |
| Phase 6 deterministic grid | **151 × 101 = 15,251** candidate operating points |
| Phase 6 production method | Differential Evolution primary; deterministic grid mandatory reference and fallback |
| Phase 7 agentic regression | **172 tests passed** after orchestration integration |
| Final project qualification | **793 tests passed** |

Important limitations remain visible in the evidence. For example, Phase 5 active-fault recall on the locked TEST set was approximately **0.678**, below the development target of `0.75`, and false alarms remained a material limitation. These observations were preserved rather than used for post-TEST retuning.

The Phase 4 timing benchmark is hardware-, software-, runtime-, implementation- and workload-dependent and must not be interpreted as an industrial throughput guarantee.

Primary evidence:

- [`docs/phase4/phase4h_summary.md`](docs/phase4/phase4h_summary.md)
- [`docs/phase5/phase5h_summary.md`](docs/phase5/phase5h_summary.md)
- [`docs/phase6/phase6g_summary.md`](docs/phase6/phase6g_summary.md)
- [`docs/phase7/phase7_summary.md`](docs/phase7/phase7_summary.md)

## 7. Agentic decision-support and authority model

Phase 7 is **deterministic decision support, not autonomous control**.

Its authority model enforces the following:

- trusted Phase 5 monitoring evidence is consumed through a narrow boundary;
- trusted Phase 6 optimisation evidence is consumed through a narrow boundary;
- boundary failures fail closed rather than inventing replacement scientific evidence;
- active faults do not automatically trigger optimisation;
- unsupported fault-to-optimisation mappings are not invented;
- operator-initiated optimisation requires an explicit request;
- recommendations require explicit human approval;
- approval events remain external to the immutable scientific decision record;
- no autonomous hardware actuation is implemented;
- no autonomous setpoint-change authority is implemented;
- an LLM has no authority to invent predictions, rewrite trusted evidence or approve physical actions.

The Phase 6 no-feasible state means only that no feasible point was found under the frozen surrogate/search protocol. It is **not** evidence of global physical infeasibility.

## 8. Repository structure

```text
.
├── artifacts/              # Persisted manifests and trained model artifacts
├── configs/                # Frozen experimental/runtime configuration
├── data/                   # Synthetic project datasets
├── docs/                   # Phase protocols, summaries, audits and closure records
│   ├── phase3/
│   ├── phase4/
│   ├── phase5/
│   ├── phase6/
│   ├── phase7/
│   └── phase8/
├── results/                # Structured frozen evaluation and qualification evidence
├── src/plasma_ai/          # Python package implementation
├── tests/                  # Unit, integration and qualification tests
├── CITATION.cff
├── LICENSE
├── RELEASE_NOTES.md
└── pyproject.toml
```

Persisted model artifacts under Phase 4 use **Git LFS**. Clone/reproduction workflows should therefore have Git LFS installed.

## 9. Installation and reproducibility

### Prerequisites

- Python `>=3.11`
- Git
- Git LFS

### Windows PowerShell

```powershell
git lfs install
git clone https://github.com/ashasif/agentic-ai-plasma-optimisation.git
cd agentic-ai-plasma-optimisation
git lfs pull

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python -m pytest -q
```

On macOS or Linux, activate the virtual environment with:

```bash
source .venv/bin/activate
```

Core package dependencies are:

- `numpy>=1.26`
- `scipy>=1.12`
- `scikit-learn>=1.5`

Development/test dependency:

- `pytest>=8.0`

The repository contains frozen manifests, structured result artifacts, persisted models and extensive phase documentation so that the evidence chain can be inspected independently of headline metrics.

## 10. Scientific scope and limitations

This project is intentionally narrower than an industrial plasma-process control system.

The following boundaries are binding:

- the scientific environment is **synthetic**;
- the plasma source is a **reduced-order argon ICP model**;
- no experimental plasma validation has been performed;
- no industrial process validation has been performed;
- no fab-scale or wafer-scale validation is claimed;
- reactive etch/deposition chemistry is not modelled or validated;
- absorbed power is **not asserted to equal generator RF power**;
- optimisation is restricted to the frozen surrogate domain;
- the Phase 6 operating domain is **15–90 W absorbed power** and **10–60 mTorr pressure**, inclusive;
- no global physical optimality is claimed;
- no global physical infeasibility is claimed;
- surrogate TEST results describe withheld synthetic data from the same modelling study;
- benchmark timing is implementation- and hardware-specific.

The project should therefore be interpreted as a reproducible research-engineering demonstration of modelling, monitoring, optimisation and governed decision-support methods—not as validation of a production plasma reactor.

## 11. Safety and governance boundaries

The system intentionally separates **scientific computation**, **decision support** and **human authority**.

The authoritative project contract prohibits:

- autonomous hardware actuation;
- autonomous setpoint changes;
- automatic conversion of a recommendation into an approved action;
- invention of physical predictions by the decision layer;
- rewriting frozen Phase 5 monitoring evidence;
- rewriting frozen Phase 6 optimisation responses;
- unsupported fault-to-optimisation mappings;
- interpreting search-protocol infeasibility as global physical infeasibility.

Every proposed operating-point recommendation remains subject to explicit human approval.

## 12. Documentation and evidence map

For detailed audit evidence, start with:

| Resource | Purpose |
|---|---|
| [`docs/project_final_summary.md`](docs/project_final_summary.md) | Whole-project technical summary and closure record |
| [`docs/project_closure_checklist.md`](docs/project_closure_checklist.md) | Final project closure evidence |
| [`docs/phase4/phase4h_summary.md`](docs/phase4/phase4h_summary.md) | Final surrogate modelling audit and evidence |
| [`docs/phase5/phase5h_summary.md`](docs/phase5/phase5h_summary.md) | Final monitoring/fault-diagnosis audit |
| [`docs/phase6/phase6g_summary.md`](docs/phase6/phase6g_summary.md) | Final optimisation audit |
| [`docs/phase7/phase7_summary.md`](docs/phase7/phase7_summary.md) | Agentic decision-support architecture and qualification |
| [`docs/phase8/phase8a_release_manifest.md`](docs/phase8/phase8a_release_manifest.md) | Post-closure release-packaging manifest |
| [`RELEASE_NOTES.md`](RELEASE_NOTES.md) | Release scope and scientific boundaries |
| [`CITATION.cff`](CITATION.cff) | Citation metadata |

Structured evidence is stored under [`results/`](results/) and persisted runtime/model artifacts under [`artifacts/`](artifacts/).

## 13. Citation

If this repository is used in academic or technical work, please use the metadata in [`CITATION.cff`](CITATION.cff).

Current software metadata identifies:

**Md Ashraful Alam Gazi**<br>
*Agentic AI for Reduced-Order ICP Plasma Process Monitoring, Fault Diagnosis and Constrained Operating-Point Optimisation*<br>
Version `0.1.0`

## 14. License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for the full licence text.
