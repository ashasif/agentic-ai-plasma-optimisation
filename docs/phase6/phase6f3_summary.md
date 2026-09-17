# Phase 6F3 ? Production Runtime Persistence and Reproducibility Qualification

## Status

Phase 6F3 passed the frozen production-runtime persistence-equivalence qualification.

## Frozen artifacts

Runtime manifest:

`artifacts/phase6/optimization_runtime_manifest.json`

Runtime-manifest SHA-256:

`5f4d7446d5c5a504c793d38cecda854b5b36b9dd65eeba6dcfe5a0c64b8d7395`

Persistence-equivalence result:

`results/phase6/runtime_persistence_equivalence.json`

Persistence-result SHA-256:

`48600b44b196d4da5e1352722c65578770e802004df50158065f8945af62ce99`

## Qualification execution

- Independent production runtime instances: 2
- Frozen persistence probes: 3
- Real Differential Evolution executions: 6
- Complete 15,251-point surrogate grid batches: 2
- Production seed: `20260914`
- Equality tolerance: `0.0`

## Probe results

### soft_density_reference

- Scenario: `soft_density_5e16`
- Runtime status: `selected_method_accepted`
- Chosen source: `differential_evolution`
- Chosen primary objective: `0.00014431717576112`
- Exact cross-instance equality: `True`
- Response SHA-256: `b50a5697cfde3904e7de1f4a2bf0d885fecc4c2bd5f38389a193ec3669736ffd`

### hard_density_min_power

- Scenario: `hard_density_1e17_min_power`
- Runtime status: `selected_method_accepted`
- Chosen source: `differential_evolution`
- Chosen primary objective: `0.3137865802302138`
- Exact cross-instance equality: `True`
- Response SHA-256: `b3abe6ad161754b7e78c68592a5191b5d93f0a43c1fbe335a6dfd25863185c3f`

### boundary_constraint_edge

- Scenario: `phase6e_boundary_margin_constraint_edge`
- Runtime status: `selected_method_accepted`
- Chosen source: `differential_evolution`
- Chosen primary objective: `0.10000000000000762`
- Exact cross-instance equality: `True`
- Response SHA-256: `0ee1fdb53ffc668702d2840ef67d876de77dbb620c15dda7aa8b81edd0ce872b`

## Reproducibility result

Every corresponding non-timing response from the two independently loaded production runtime instances was exactly identical.

Timing was recorded separately and excluded from equivalence comparison, as required by the frozen runtime protocol.

The two runtime instances independently reproduced the exact frozen grid input, density and electron-temperature array identities.

## Same-instance contract

The frozen Phase 6F1 protocol specifies three persistence probes across two independent runtime instances, giving six real optimizer executions. Same-instance exact response serialization was already qualified synthetically in Phase 6F2 and remained covered by the passing Phase 6 regression suite.

## Scientific boundaries

This qualification is limited to the frozen synthetic reduced-order argon ICP surrogate domain.

It does not establish:

- experimental validation;
- industrial validation;
- global continuous optimality;
- global physical infeasibility outside the frozen search protocol;
- wafer-scale or reactive-chemistry validity.

No Phase 4 TEST or Phase 5 TEST evidence was reopened.

No SHGO execution, optimizer retuning, source-simulator execution or agentic orchestration occurred.

## Phase progression

Phase 6F is complete subject to final repository closure verification.

Phase 6G may be unlocked after this evidence is committed.

Phase 7 remains locked until formal Phase 6 closure.
