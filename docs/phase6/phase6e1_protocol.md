# Phase 6E1 ? Robustness and Scientific-Qualification Protocol

## Status

Phase 6E1 freezes the robustness and scientific-qualification contract for the
selected Phase 6 optimizer before additional optimization is executed.

Protocol:

`configs/phase6/robustness_qualification_protocol.json`

SHA-256:

`5897dd29172a64e1192d0f40c848c36d717c6c044b8792e8d7fbc3de47a4cfda`

Selected continuous optimizer:

**Differential Evolution**

Mandatory reference/fallback:

**Deterministic 15,251-point grid**

SHGO remains closed as a non-qualified comparator and is not executed in
Phase 6E.

## Purpose

Phase 6D established that Differential Evolution is the only continuous method
that satisfies the frozen benchmark qualification contract.

Phase 6E now tests whether that selection remains defensible under additional
predeclared challenges involving:

- RNG-seed robustness;
- hard constraint-edge behavior;
- deliberate infeasibility;
- small local operating-point perturbations;
- multiobjective tradeoff structure.

No optimizer parameter retuning is allowed.

## Seed robustness

The frozen seeds are:

1. `20260914` ? production seed;
2. `20260915` ? qualification-only alternate seed;
3. `20260916` ? qualification-only alternate seed.

Changing the seed is a robustness diagnostic, not production retuning.

All other Differential Evolution parameters remain exactly those frozen in
Phase 6D.

Exact decision-coordinate equality across different seeds is not required.

For expected-feasible challenges:

- all three runs must pass explicit Phase 6 feasibility;
- objective spread must not exceed `1e-6`;
- each objective must be no worse than the deterministic-grid reference by
  more than `1e-6`.

## Qualification challenges

### 1. Selected-method replay

The already-frozen:

`hard_density_1e17_min_power`

scenario is rerun unchanged under all three Phase 6E seeds.

This checks that the Phase 6D selection remains robust to predeclared seed
variation.

### 2. Boundary-margin constraint edge

A new predeclared geometric challenge uses:

- minimum normalized boundary margin: `0.10`;
- objective: minimize absorbed power.

The geometry itself implies:

- minimum feasible power: `22.5 W`;
- maximum feasible power: `82.5 W`;
- feasible pressure interval: `15?55 mTorr`;
- theoretical minimum normalized absorbed-power objective: `0.10`.

This isolates enforcement of a hard constraint edge without relying on a
specific plasma target.

### 3. Deliberately infeasible high-density target

The hard target is frozen at:

`2.8626132708953632e+17 m^-3`

This is exactly `1.25` times the frozen deterministic-grid maximum density.

The allowed relative density error is only `1%`.

At the frozen grid maximum density, the relative error to this target is
`20%`.

The challenge is deliberately constructed to test fail-closed infeasibility
handling.

A no-feasible result is interpreted only as:

`no_feasible_point_found_under_search_protocol`

It is not evidence that such a state is physically impossible.

## Planned selected-method execution

The challenge suite contains:

- 3 scenarios;
- 3 seeds;
- 9 Differential Evolution runs.

SHGO is not rerun.

## Mandatory deterministic reference

One shared 15,251-point production-surrogate batch may be used for:

- deterministic references for all three challenges;
- tradeoff diagnostics for the five deferred multiobjective scenarios.

No second complete-grid surrogate batch is allowed during the Phase 6E
qualification execution.

## Local perturbation diagnostics

The representative Differential Evolution candidate from each of the five
Phase 6D qualification scenarios is examined using the eight offsets formed by:

- power ? `0.25 W`;
- pressure ? `0.25 mTorr`;
- all four diagonal combinations.

Coordinates are never clipped.

A perturbation outside the qualified domain is recorded as not evaluated.

Valid perturbations record:

- predicted electron density;
- predicted electron temperature;
- scenario feasibility;
- active constraint values;
- primary objective;
- objective delta;
- normalized boundary margin.

Not every neighbour is required to remain feasible. An optimum located on a
hard constraint edge can legitimately be locally sensitive.

## Multiobjective tradeoff diagnostics

Continuous multiobjective optimization remains prohibited because no
continuous scalarization preference has been frozen.

The five deferred multiobjective Phase 6C scenarios are therefore analysed
using exact deterministic-grid Pareto semantics only.

For every Pareto set Phase 6E records:

- feasible count;
- Pareto count;
- objective ranges;
- power and pressure ranges;
- predicted density and temperature ranges;
- whether the frozen lexicographic point lies on the Pareto set.

No weighted scalarization is introduced.

No knee point is selected.

No production operating point is selected from these diagnostics.

## Phase 6E pass condition

Phase 6E passes only if:

1. selected-method seed robustness passes;
2. deliberate infeasibility is handled without a false feasible result;
3. all perturbation diagnostics are completed without clipping or
   out-of-domain inference;
4. all five grid-based tradeoff diagnostics are completed under exact Pareto
   semantics;
5. all upstream frozen evidence remains unchanged.

Failure does not authorize retuning.

A failed gate must stop for diagnosis and any redevelopment would require a
separately frozen amendment.

## Scientific boundaries

Phase 6E remains qualification of optimization against the frozen reduced-order
surrogate.

It does not claim:

- experimental validation;
- industrial validation;
- OIPT hardware validation;
- a mathematically proven global continuous optimum;
- physical impossibility from an infeasible optimization challenge.

## Phase progression

Phase 6F remains locked until Phase 6E passes.

Phase 6G and Phase 7 also remain locked.
