# Phase 5F Closure Checklist

## Frozen development state

- [x] Phase 5 protocol remained frozen.
- [x] Phase 5E-R protocol remained frozen.
- [x] Redevelopment benchmark evidence remained frozen.
- [x] Redevelopment selection remained frozen.
- [x] Final TRAIN+VALIDATION refit remained frozen.
- [x] Detector remained `ERD4_extra_trees_threshold_040`.
- [x] Detector threshold remained `0.40`.
- [x] Diagnoser remained `ERG3_extra_trees`.
- [x] TEST rows used for model fitting remained `0`.

## Evaluator governance

- [x] Dedicated Phase 5F evaluator implemented before TEST use.
- [x] Evaluator qualified using synthetic fixtures.
- [x] Evaluator implementation frozen before real TEST use.
- [x] Real TEST result absent before one-time execution.
- [x] TEST access required the dedicated explicit unlock path.
- [x] Writer refused overwrite of existing TEST evidence.

## One-time TEST evaluation

- [x] TEST features were interpreted only during Phase 5F.
- [x] TEST targets were interpreted only during Phase 5F.
- [x] TEST predictive evaluation was executed exactly once.
- [x] TEST row count was `1024`.
- [x] TEST episode count was `16`.
- [x] TEST predictions were generated.
- [x] TEST performance was calculated.
- [x] TEST holdout is permanently consumed.
- [x] No second evaluation was executed.

## Frozen TEST evidence

- [x] `results/phase5/locked_test_evaluation.json` exists.
- [x] Frozen TEST evidence SHA-256 is `63ff1ea3448e0ab49ef0ea13f3e4cfbdcea985b7ede624b0d61cb17d5da53fb3`.
- [x] TEST evidence was committed independently.
- [x] Evidence-freeze commit is `3e6e357c57360c45e5a17a4fefde2a0c3c8da773`.
- [x] Frozen upstream artifacts remained unchanged.
- [x] Full repository regression passed after TEST evaluation.

## Final TEST interpretation

- [x] Detector balanced accuracy recorded.
- [x] Detector macro F1 recorded.
- [x] Detector active recall recorded.
- [x] Detector precision and specificity recorded.
- [x] AUROC and average precision recorded.
- [x] Per-fault-family detector recall recorded.
- [x] Sequence diagnostics recorded.
- [x] All 12 fault episodes were eventually detected.
- [x] Detection delays are explicitly reported in ordered steps, not physical time.
- [x] Pre-active false alarms are documented.
- [x] Normal-episode false alarms are documented.
- [x] Diagnoser macro F1 recorded.
- [x] Diagnoser balanced accuracy recorded.
- [x] Per-class diagnostic performance recorded.
- [x] Majority-baseline improvement recorded.
- [x] Secondary end-to-end performance recorded.
- [x] End-to-end pressure-path weakness documented.

## Scientific limitations

- [x] Synthetic-data scope is explicit.
- [x] Reduced-order argon-model scope is explicit.
- [x] No experimental validation claim is made.
- [x] No industrial validation claim is made.
- [x] No OIPT operating-range claim is made.
- [x] Pressure-path diagnosis remains mechanism-ambiguous.
- [x] TEST detector active recall below the development threshold is documented rather than tuned away.
- [x] False-alarm limitations are documented.
- [x] TEST evidence is not treated as development feedback.

## Post-TEST governance

- [x] TEST-based model selection is prohibited.
- [x] TEST-based threshold selection is prohibited.
- [x] TEST-based retuning is prohibited.
- [x] TEST-based candidate changes are prohibited.
- [x] TEST reruns for performance improvement are prohibited.
- [x] Phase 5G may use only the frozen detector and diagnoser.
- [x] Phase 5G may not reopen model development from Phase 5F evidence.

## Closure

- [x] Phase 5F computational work complete.
- [x] Phase 5F evidence frozen.
- [x] Phase 5F governance complete.
- [x] Phase 5F scientific limitations documented.
- [x] Phase 5F formally ready for closure.
- [x] Phase 5G may be unlocked.
