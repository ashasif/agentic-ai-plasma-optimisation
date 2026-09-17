# Phase 6F2.1 ? Runtime Manifest Validation Hardening

## Status

This gate hardens the Phase 6 production-runtime manifest loader before the
official Phase 6F3 runtime manifest is created.

The frozen Phase 6F1 and Phase 6F1.1 runtime protocols remain unchanged.

## Reason

The Phase 6F2 manifest payload already records:

- continuous benchmark base SHA-256;
- continuous benchmark amendment SHA-256;
- grid input-array SHA-256;
- grid density-array SHA-256;
- grid temperature-array SHA-256.

The loader must fail closed if any of those identities drift.

## Added validation

The runtime-manifest loader now validates:

1. runtime-manifest schema version;
2. mandatory reference method;
3. fallback method;
4. frozen grid input-array identity;
5. frozen grid density-array identity;
6. frozen grid temperature-array identity;
7. continuous benchmark base file SHA-256;
8. continuous benchmark amendment file SHA-256.

Existing fail-closed validation remains in place for:

- runtime base protocol SHA-256;
- runtime amendment SHA-256;
- effective runtime identity;
- five-state runtime vocabulary;
- production seed;
- selected optimizer;
- Phase 4G surrogate manifest;
- density surrogate;
- temperature surrogate;
- optimizer-selection record;
- Phase 6E qualification evidence;
- continuous benchmark result;
- deterministic-grid result;
- runtime implementation source.

## Scientific scope

This is implementation hardening only.

It changes no:

- objective;
- constraint;
- optimizer;
- optimizer parameter;
- production seed;
- deterministic grid;
- fallback state;
- robustness result;
- scientific conclusion.

No production surrogate is loaded.

No optimization is executed.

## Qualification

Synthetic negative tests deliberately tamper each newly protected manifest
field and require fail-closed rejection.

The complete Phase 6 optimisation regression suite must also pass.

## Next step

After this gate passes, Phase 6F3 may:

- create and freeze the official runtime manifest;
- construct two independent production runtime instances;
- run all three frozen persistence probes in each instance;
- compare all corresponding non-timing outputs exactly;
- freeze persistence-equivalence evidence.

Phase 6G and Phase 7 remain locked.
