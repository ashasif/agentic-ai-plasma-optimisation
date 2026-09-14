# Phase 6E2 — Robustness Qualification Engine and Synthetic Qualification

## Status

Phase 6E2 implements the machinery required by the frozen Phase 6E1
robustness-qualification protocol.

This gate creates engineering qualification evidence only.

It does not load or evaluate the production Phase 4G surrogate.

## Frozen protocol

`configs/phase6/robustness_qualification_protocol.json`

SHA-256:

`5897dd29172a64e1192d0f40c848c36d717c6c044b8792e8d7fbc3de47a4cfda`

## Selected-method robustness runner

The engine supports only the selected:

`differential_evolution`

method.

The only permitted Phase 6E variation is the predeclared RNG seed:

- `20260914`
- `20260915`
- `20260916`

Every other Differential Evolution parameter is read unchanged from the frozen
Phase 6D benchmark contract.

Arbitrary seeds are rejected.

SHGO is not implemented in the Phase 6E robustness runner.

## Final feasibility

Each selected-method result is independently re-evaluated using the frozen
Phase 6 primitives.

A run passes only when:

- the final coordinates are in the qualified domain;
- the physical predictions are finite and strictly positive;
- the scenario is explicitly feasible;
- every active `g(x) >= 0` constraint passes;
- the frozen primary objective is finite.

The SciPy success flag remains separate from this scientific validity check.

## Infeasibility handling

The engine allows an optimizer to return a numerical candidate for an
infeasible scenario, but that candidate is not silently accepted.

The explicit Phase 6 feasibility check determines whether a feasible point was
actually returned.

Synthetic qualification verifies that the deliberately unreachable Phase 6E
high-density challenge does not produce a false feasible result under any of
the three frozen seeds.

## Perturbation diagnostics

The engine evaluates exactly the eight Phase 6E perturbation offsets.

No coordinate clipping is performed.

Out-of-domain neighbours are explicitly marked as not evaluated.

For in-domain neighbours the engine records:

- physical predictions;
- scenario feasibility;
- active constraints;
- primary objective;
- objective delta;
- normalized boundary margin.

## Grid references

The engine can evaluate arbitrary Phase 6E challenge scenarios using the
already-defined deterministic-grid contract and supplied prediction arrays.

It does not perform surrogate loading itself.

This preserves the Phase 6E requirement that the real execution gate may use
one shared 15,251-point production-surrogate prediction batch.

## Multiobjective tradeoff summaries

The engine uses the exact deterministic-grid Pareto set produced by the Phase
6 grid machinery.

For each multiobjective scenario it summarizes:

- feasible count;
- Pareto count;
- frozen objective-component ranges;
- power range;
- pressure range;
- electron-density range;
- electron-temperature range;
- whether the frozen lexicographic selection belongs to the Pareto set.

No weighted scalarization is introduced.

No continuous multiobjective optimizer is executed.

No knee point or new production operating point is selected.

## Synthetic qualification

Tests use only the development predictor:

- density = `2.5e15 * absorbed_power_W`
- Te = `1.55 + 0.005 * pressure_mTorr`

The synthetic tests cover:

- exact Phase 6E protocol identity;
- exact seed schedule;
- exact challenge ordering;
- rejection of unapproved seeds;
- selected DE execution across all three seeds;
- objective stability at the geometric boundary challenge;
- deliberate infeasibility without false feasibility;
- deterministic-grid challenge evaluation;
- no-clipping perturbation accounting;
- interior perturbation evaluation;
- exact grid Pareto summary;
- rejection of single-objective input to the tradeoff summarizer.

## Explicit exclusions

This gate performs no:

- Phase 4G production-surrogate loading;
- real Phase 6E optimization;
- real perturbation inference;
- real Pareto/tradeoff inference;
- SHGO execution;
- optimizer retuning;
- source-simulator execution;
- Phase 4 TEST access;
- Phase 5 TEST access.

## Next step

After this implementation is committed, Phase 6E3 may execute the real frozen
robustness qualification.

That execution will generate:

- nine selected-method DE runs;
- one shared 15,251-point surrogate grid batch;
- three grid challenge references;
- five source-candidate perturbation diagnostics;
- five exact grid Pareto/tradeoff summaries.

Phase 6F remains locked until that evidence passes the frozen Phase 6E
qualification contract.
