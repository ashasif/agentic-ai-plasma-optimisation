# Phase 6B2 — Trusted Surrogate Adapter and Prediction-Contract Qualification

## Purpose

Phase 6B2 connects the controlled Phase 6 optimisation primitives to the
already frozen Phase 4G production surrogate.

This is an inference-interface qualification gate, not an optimisation gate.

## Trusted inference path

Phase 6 uses:

`plasma_ai.surrogate.phase4g_persistence.load_phase4g_surrogate`

and then:

`LoadedPhase4GSurrogate.predict_physical`

No Phase 6 code directly deserializes the density or temperature model files.

The frozen Phase 4G loader remains responsible for validating the production
manifest, exact artifact bytes, estimator contracts and runtime compatibility.

## Phase 6 adapter behaviour

The Phase 6 adapter:

- loads the effective Phase 6 protocol first;
- verifies the frozen Phase 4G manifest identity;
- verifies the frozen Phase 4G persistence-source identity;
- preserves feature order:
  1. `nominal_absorbed_power_W`
  2. `target_pressure_mTorr`;
- validates every operating point before surrogate inference;
- rejects out-of-domain inputs rather than clipping them;
- requires prediction vectors to have the exact batch shape;
- requires all physical predictions to be finite and strictly positive;
- returns independent read-only prediction arrays.

## Qualification strategy

Phase 6B2 qualification uses a small predeclared fixed set of operating
points spanning:

- the four qualified-domain corners;
- the qualified-domain centre;
- four interior representative points.

The same fixed batch is evaluated repeatedly and pointwise to establish:

- successful trusted loading;
- valid physical prediction contract;
- deterministic repeatability;
- batch/point inference equivalence.

These checks are inference-contract evidence only.

They are not:

- optimisation-performance evidence;
- source-model validation;
- experimental validation;
- industrial validation;
- OIPT hardware validation.

## Explicit exclusions

Phase 6B2 performs no:

- deterministic optimisation grid search;
- differential evolution;
- SHGO;
- local numerical optimisation;
- optimiser-method selection;
- source-simulator execution;
- Phase 4 TEST-target access;
- Phase 5 TEST-dataset access;
- Phase 6 result artifact generation.

## Next step

After Phase 6B2 closes, Phase 6C may introduce the mandatory deterministic
15,251-point optimisation grid baseline under the already frozen protocol.
