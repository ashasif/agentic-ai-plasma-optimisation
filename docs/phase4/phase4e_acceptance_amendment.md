# Phase 4E Pre-Test Acceptance Amendment

**Status:** APPROVED BEFORE PHASE 4E PROBE EVALUATION
**Approval date:** 2026-09-12
**Amendment type:** Pre-test protocol clarification

## 1. Purpose

The frozen Phase 4A protocol defines the Phase 4E physics-aware checks,
the 41 x 41 Cartesian probe grid, source-simulator-first evaluation,
and the requirement that surrogate structural behaviour be interpreted
relative to the source reduced-order simulator.

However, the frozen protocol does not assign deterministic numerical
PASS/FAIL semantics to unsupported monotonic-trend violations or to
the phrase "obviously spurious local oscillation".

This amendment freezes those missing decision semantics before any
Phase 4E probe-grid results are observed.

The Phase 4A protocol itself is not rewritten or silently modified.

## 2. Source-reference gate

The Phase 4E probe design must contain exactly 1,681 unique points.

The canonical Phase 3 source-simulator path must be evaluated before
the selected surrogate candidates are evaluated.

Before surrogate acceptance is attempted:

- all source target values must be finite;
- every source row must be qualification-valid;
- the probe grid must contain exactly 1,681 unique points.

If this source-reference gate fails, Phase 4E stops and the condition is
reported as a source-reference grid failure, not as surrogate failure.

## 3. Positivity rules

For every Phase 4E probe point:

- predicted electron density must be finite and strictly positive;
- predicted electron temperature must be finite and strictly positive.

Any violation fails the corresponding target candidate.

## 4. Source-referenced trend rule

For two adjacent points along a frozen diagnostic grid line:

`source_delta = source[i+1] - source[i]`

`surrogate_delta = surrogate[i+1] - surrogate[i]`

An unsupported trend reversal occurs only when:

- `source_delta > 0` and `surrogate_delta < 0`; or
- `source_delta < 0` and `surrogate_delta > 0`.

If the source delta is exactly zero, that interval imposes no directional
requirement.

If the surrogate delta is exactly zero, it is not counted as an opposite
direction reversal.

No empirical magnitude tolerance is introduced.

The permitted number of unsupported trend reversals is zero.

This rule applies to:

1. electron density versus absorbed power at fixed pressure;
2. electron density versus pressure at fixed absorbed power;
3. electron temperature versus pressure at fixed absorbed power.

## 5. Local spurious-oscillation rule

At an interior point of a diagnostic grid line, define the two adjacent
increments.

A strict local turning point exists when:

`delta_left * delta_right < 0`

A zero increment therefore does not form a strict turning point.

A surrogate turning point is classified as spurious only when the
surrogate has a strict turning point at an interior grid location and
the source simulator does not have a strict turning point at that same
location.

The permitted number of source-unsupported surrogate turning points is
zero.

This check applies to:

- density along absorbed power at fixed pressure;
- density along pressure at fixed absorbed power;
- temperature along pressure at fixed absorbed power.

No temperature-versus-power criterion is added because it is not one of
the frozen Phase 4A structural trend checks.

## 6. PASS / FAIL rule

A target candidate passes Phase 4E acceptance only if every applicable
mandatory check passes.

There is no post-result threshold adjustment.

If either Phase 4D-selected target candidate fails:

- stop Phase 4E;
- do not silently select another Phase 4D candidate;
- do not retune;
- do not modify the target transformation;
- do not change the acceptance criteria after observing the result.

Both selected target candidates must pass before final refitting is
permitted.

## 7. Final refit gate

Only after Phase 4E acceptance passes:

- rebuild the exact Phase 4D-selected configurations;
- retain the selected target transformations;
- combine TRAIN and VALIDATION only;
- refit using exactly 6,144 rows;
- perform no retuning or model reselection;
- keep TEST targets locked.

The refitted objects remain pre-test locked surrogate candidates.
Production persistence belongs to Phase 4G.

## 8. Scientific boundary

Phase 4E remains structured in-envelope physics-aware behavioural
validation of a surrogate of the synthetic reduced-order argon plasma
model.

It is not:

- OOD validation;
- experimental validation;
- industrial validation;
- an OIPT operating-range claim;
- reactive etch/deposition prediction;
- wafer-scale spatial modelling.

Absorbed power is not generator RF power.
