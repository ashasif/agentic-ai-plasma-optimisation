# Phase 4F — One-Time Locked TEST Evaluation Summary

## Status

**Phase:** 4F
**Technical status:** FORMALLY CLOSED
**TEST evaluation:** COMPLETED ONCE
**TEST status:** CONSUMED - MUST NOT BE REUSED FOR DEVELOPMENT
**Retuning after TEST:** PROHIBITED
**Phase 4G status:** UNLOCKED

---

## 1. Purpose

Phase 4F performed the single predeclared locked TEST evaluation of the final
surrogate configuration frozen before TEST access.

The phase was designed to answer one question:

> How well does the completely frozen surrogate generalise to the untouched
> 2,048-row TEST split?

Phase 4F was not a development or tuning phase.

No TEST result may be used for:

- model reselection;
- hyperparameter tuning;
- target-transform changes;
- feature changes;
- acceptance-rule changes;
- redevelopment of the Phase 4 surrogate.

---

## 2. Pre-TEST state

Phase 4F began only after Phase 4E-R had been formally closed.

Formal Phase 4E-R closure commit:

`0f604d7`

Frozen Phase 4F execution protocol commit:

`6a20561`

Pre-TEST analysis implementation commit:

`59f4780`

Post-refit structural diagnostics commit:

`f4421ef`

Locked TEST evaluator implementation commit:

`8c2ab75`

Before TEST access:

- final surrogate configurations were frozen;
- physics-aware acceptance had passed;
- final TRAIN + VALIDATION refit was complete;
- final fit rows were exactly 6,144;
- TEST targets had never been accessed;
- Phase 4F targeted regression passed;
- full repository regression passed;
- repository working tree was clean.

---

## 3. Frozen data-use contract

Frozen dataset:

`data/synthetic/phase3/base_steady_state.csv`

Frozen split sizes:

| Split | Rows |
|---|---:|
| TRAIN | 4,096 |
| VALIDATION | 2,048 |
| TEST | 2,048 |

Final surrogate fit:

`TRAIN + VALIDATION = 6,144 rows`

Final evaluation:

`TEST = 2,048 rows`

The TEST split was opened exactly for the Phase 4F locked final evaluation.

---

## 4. Frozen feature contract

Exact feature order:

1. `nominal_absorbed_power_W`
2. `target_pressure_mTorr`

No additional feature was introduced.

---

## 5. Final density surrogate

Target:

`true_electron_density_m3`

Model:

`HistGradientBoostingRegressor`

Transform:

`log10`

Frozen configuration:

- learning rate: `0.05`
- maximum iterations: `400`
- maximum leaf nodes: `31`
- L2 regularisation: `0.1`
- monotonic constraints: `[1, 1]`
- early stopping: `False`
- random state: `20260913`

The monotonic constraints encode measured structure of the frozen reduced-order
source model within this project envelope. They are not claimed as universal
experimental ICP laws.

---

## 6. Final temperature surrogate

Target:

`true_electron_temperature_eV`

Model:

`ExtraTreesRegressor`

Transform:

`identity`

Frozen configuration:

- number of estimators: `500`
- maximum depth: `None`
- minimum samples per leaf: `1`
- maximum features: `1.0`
- random state: `20260913`

---

## 7. Pre-TEST verification

Immediately before the irreversible TEST evaluation:

Phase 4F targeted regression:

`22 passed`

Complete repository regression:

`381 passed`

The real Phase 4F result artifact did not yet exist.

The working tree was clean.

TEST targets were still unopened at that checkpoint.

---

## 8. One-time TEST evaluation

The real locked TEST evaluation was then performed once.

TEST rows evaluated:

`2,048`

The evaluation recorded:

- TEST targets accessed: `True`;
- TEST evaluation performed: `True`;
- TEST may affect model selection: `False`;
- retuning after TEST permitted: `False`.

The TEST set is therefore considered consumed for Phase 4 surrogate
development.

---

## 9. Density TEST performance

Final density metrics:

| Metric | TEST result |
|---|---:|
| MAE | 1234386463622997.5 m^-3 |
| RMSE | 1682635059428192.2 m^-3 |
| R2 | 0.9987458341401417 |
| Mean absolute relative error | 0.01373138859593015 |
| Median absolute relative error | 0.010308964790501812 |
| P95 absolute relative error | 0.03860904857730792 |
| Maximum absolute relative error | 0.0848424039046155 |

For readability:

- mean absolute relative error: approximately `1.37%`;
- median absolute relative error: approximately `1.03%`;
- P95 absolute relative error: approximately `3.86%`;
- maximum absolute relative error: approximately `8.48%`.

These values are final TEST evidence and are not model-selection criteria.

---

## 10. Temperature TEST performance

Final temperature metrics:

| Metric | TEST result |
|---|---:|
| MAE | 6.387538320240006e-06 eV |
| RMSE | 1.0578939071288781e-05 eV |
| R2 | 0.9999999939303662 |
| Mean absolute relative error | 3.5625681248898348e-06 |
| Median absolute relative error | 2.124978678250491e-06 |
| P95 absolute error | 2.2230912613851984e-05 eV |
| Maximum absolute error | 0.00010603452518287426 eV |

These values are final TEST evidence and are not development feedback.

---

## 11. Prediction integrity

### Density

Prediction count:

`2,048`

All predictions finite:

`True`

All predictions strictly positive:

`True`

Prediction minimum:

`1.6197742107743246e16 m^-3`

Prediction maximum:

`2.2700130940788067e17 m^-3`

### Temperature

Prediction count:

`2,048`

All predictions finite:

`True`

All predictions strictly positive:

`True`

Prediction minimum:

`1.5473396132126394 eV`

Prediction maximum:

`2.0740648030208093 eV`

---

## 12. Error-space analysis

The Phase 4F result artifact contains the full frozen error-space analysis for
both targets.

The analysis uses physical-domain definitions frozen before TEST access.

Reported regions include:

- near-boundary;
- interior;
- central domain;
- low, middle, and high power;
- low, middle, and high pressure;
- low-power / low-pressure corner;
- low-power / high-pressure corner;
- high-power / low-pressure corner;
- high-power / high-pressure corner.

The Phase 4 near-boundary threshold is based on the frozen `10%` normalized
physical-domain rule.

The Phase 3 `domain_status` labels were not reused for Phase 4F boundary
analysis.

All error-space region contracts passed factual QA.

---

## 13. Worst-case observations

The result artifact records exactly:

- 10 worst density TEST observations;
- 10 worst temperature TEST observations.

Frozen ranking rules:

### Density

Descending absolute relative error.

### Temperature

Descending absolute error in eV.

Exact ties are resolved using original TEST split index ascending.

These observations are descriptive final-evaluation evidence only.

They must not trigger surrogate modification.

---

## 14. Post-refit structural diagnostics

The final 6,144-row-refitted surrogate was evaluated on the already-frozen:

`41 x 41`

physics probe grid.

Total probe points:

`1,681`

The existing Phase 4E-R source-reference rules were reused.

Final results:

- density structural diagnostic: `PASS`;
- temperature structural diagnostic: `PASS`;
- overall structural diagnostic: `PASS`.

These diagnostics are post-refit reporting evidence rather than a new
selection gate.

---

## 15. Frozen TEST result artifact

Artifact:

`results/phase4/locked_test_evaluation.json`

SHA-256:

`a14638013561b8f1f66547fab9910d96b0a2ad2c12afad894948b1f690af14e0`

Artifact freeze commit:

`19e2dd7`

Artifact size in Git:

`694 inserted lines`

---

## 16. Result factual QA

The frozen Phase 4F TEST artifact passed:

`48 / 48`

factual QA checks.

The QA verified, among other things:

- artifact SHA-256;
- Phase 4F stage identity;
- irreversible-evaluation state;
- 6,144 final fit rows;
- 2,048 TEST rows;
- TEST access state;
- no-retuning state;
- metric finiteness;
- prediction integrity;
- 10 worst observations per target;
- full error-space region contract;
- boundary/interior partition totals;
- power-third partition totals;
- pressure-third partition totals;
- density structural PASS;
- temperature structural PASS;
- overall structural PASS;
- TEST treated as final evidence;
- TEST explicitly excluded from development feedback.

---

## 17. Interpretation

The Phase 4F results provide strong synthetic simulator-surrogate
generalisation evidence within the frozen numerically qualified model envelope.

The density surrogate retains high predictive accuracy on the untouched TEST
split while satisfying the frozen structural diagnostics.

The temperature surrogate shows extremely small TEST error relative to the
variation present in the reduced-order simulator output.

These results must not be interpreted as experimental plasma validation.

---

## 18. Scientific claim boundary

All Phase 4F interpretation remains subject to the project claim boundary:

- the data are synthetic;
- the source is a reduced-order argon plasma model;
- the operating envelope is numerically qualified;
- the surrogate approximates the reduced-order simulator;
- there is no experimental validation;
- there is no industrial validation;
- there is no OIPT operating-range claim;
- absorbed power is not generator RF power;
- reactive etch or deposition behaviour is not predicted;
- wafer-scale spatial behaviour is not modelled.

The TEST results therefore demonstrate generalisation to withheld synthetic
simulator data, not industrial process performance.

---

## 19. Irreversibility

The Phase 4 TEST split has now been consumed.

The following are prohibited:

- rerunning Phase 4 surrogate development against TEST;
- using TEST residuals to retune parameters;
- replacing the selected model because of TEST performance;
- changing the target transform from TEST evidence;
- changing the feature set from TEST evidence;
- modifying physics-acceptance rules from TEST evidence;
- treating TEST as a new validation split.

The frozen TEST result must be preserved as final Phase 4 evaluation evidence.

---

## 20. Phase boundary

Phase 4F is formally closed.

All required closure evidence is complete:

- one-time locked TEST evaluation completed;
- frozen TEST artifact committed at `19e2dd7`;
- TEST result factual QA passed: 48 / 48 checks;
- Phase 4F closure documentation factual QA passed: 61 / 61 checks;
- Phase 4F closure documentation committed at `81eb713`;
- clean working tree confirmed before formal closure;
- TEST is permanently consumed for Phase 4 surrogate development;
- retuning from TEST remains prohibited.

Phase 4G is formally unlocked.

Phase 4G must use the already-frozen surrogate configuration and must not
reopen surrogate development based on Phase 4F TEST performance.

No further Phase 4 TEST evaluation is permitted.
