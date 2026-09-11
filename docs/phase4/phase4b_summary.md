# Phase 4B - Dataset, Transformations, Metrics and Reference Baselines

**Status:** COMPLETE - VALIDATED AND READY FOR CLOSURE COMMIT

## 1. Scope

Phase 4B established the controlled modelling infrastructure required before the
classical surrogate benchmark.

No final surrogate model was selected in Phase 4B.

No TEST target values were accessed.

The work in this subphase used the frozen Phase 3 base dataset and the frozen
Phase 4A modelling protocol.

## 2. Dataset contract

Source dataset:

`data/synthetic/phase3/base_steady_state.csv`

Frozen SHA-256:

`dc8d3a24f84b3eca9ce930a88e969e51892d69430ce5840cc546523e9f9dd423`

Rows:

- TRAIN: 4,096
- VALIDATION: 2,048
- TEST: 2,048
- Total: 8,192

Approved features:

- `nominal_absorbed_power_W`
- `target_pressure_mTorr`

Primary targets:

- `true_electron_density_m3`
- `true_electron_temperature_eV`

The Phase 4 loader verifies the dataset hash, required columns, split counts,
finite values, and frozen feature/target ordering.

TEST features may be loaded for structural checks, but TEST targets are withheld
by default.

## 3. Dependencies

Phase 4 introduced:

- scikit-learn

The modelling loader itself does not require pandas.

Current validated environment during Phase 4B development:

- Python 3.12.5
- NumPy 2.5.3
- SciPy 1.18.1
- scikit-learn 1.9.1
- joblib 1.6.0

## 4. Target transformations

The frozen Phase 4A transformation policy was implemented.

Electron density candidates:

- identity
- log10

Electron temperature candidate:

- identity

The log10 transformation requires strictly positive values.

All predictions from transformed-target models are inverse transformed to the
physical target scale before evaluation.

No transformation has yet been selected as the final electron-density
representation.

## 5. Evaluation metrics

Electron-density metrics:

- MAE
- RMSE
- R-squared
- mean absolute relative error
- median absolute relative error
- P95 absolute relative error
- maximum absolute relative error

Electron-temperature metrics:

- MAE in eV
- RMSE in eV
- R-squared
- mean absolute relative error
- median absolute relative error
- P95 absolute error in eV
- maximum absolute error in eV

The frozen primary validation discriminator for electron density is P95 absolute
relative error.

The frozen primary validation discriminator for electron temperature is RMSE in
eV.

## 6. Reference baseline experiment

Structured result artifact:

`results/phase4/reference_baselines.json`

Artifact SHA-256 at Phase 4B evaluation:

`33ab210590ccd4d54d385ea37b356b5d1b8c0d7ad494088219072cd31b465341`

Data use:

- fitting: TRAIN only
- evaluation: VALIDATION only
- TEST targets accessed: no

Six frozen reference experiments were run.

### 6.1 Electron density

#### Dummy mean - identity

- R-squared: approximately 0
- median absolute relative error: 34.58%
- P95 absolute relative error: 196.27%

This provides the non-informative reference baseline.

#### Dummy mean - log10

- R-squared: -0.0635
- median absolute relative error: 36.56%
- P95 absolute relative error: 161.50%

The inverse-transformed mean in log space is not equivalent to the arithmetic
mean in physical density space, so this baseline is not expected to reproduce
the identity dummy baseline.

#### Linear regression - identity

- R-squared: 0.96418
- RMSE: approximately 8.99e15 m^-3
- median absolute relative error: 4.75%
- P95 absolute relative error: 36.73%
- maximum absolute relative error: 115.54%

#### Linear regression - log10

- R-squared on physical density scale: 0.93664
- RMSE on physical density scale: approximately 1.20e16 m^-3
- median absolute relative error: 8.60%
- P95 absolute relative error: 25.05%
- maximum absolute relative error: 57.73%

The density baseline therefore exposes a meaningful trade-off.

The identity linear model performs better on global squared-error measures,
whereas the log10 linear model gives better high-percentile and worst-case
relative error.

Phase 4B does not resolve this trade-off.

Both density representations must remain eligible for the Phase 4C/4D
validation benchmark.

### 6.2 Electron temperature

#### Dummy mean - identity

- R-squared: approximately 0
- RMSE: 0.13579 eV
- P95 absolute error: 0.27770 eV

#### Linear regression - identity

- R-squared: 0.91813
- RMSE: 0.03885 eV
- MAE: 0.03177 eV
- P95 absolute error: 0.07494 eV
- maximum absolute error: 0.13541 eV

The simple linear baseline therefore captures a substantial fraction of the
structured temperature response while leaving measurable nonlinear error for
the Phase 4C candidate models to address.

## 7. Interpretation

The reference results show that the reduced-order simulator response is
substantially learnable from the two approved inputs.

They do not establish that a linear model is the final surrogate.

They also do not establish that either identity or log10 density representation
is final.

Those choices remain governed by the frozen Phase 4 validation procedure.

The baseline results are comparisons against synthetic outputs from the
reduced-order argon plasma simulator.

They are not experimental validation and are not evidence of industrial OIPT
accuracy.

## 8. Scientific claim boundary

Phase 4B continues to preserve the following limitations:

- synthetic data;
- reduced-order argon plasma model;
- numerically qualified model envelope;
- surrogate of the reduced-order simulator;
- no experimental validation;
- no industrial validation;
- no OIPT operating-range claim;
- absorbed power is not generator RF power;
- no reactive etch/deposition prediction;
- no wafer-scale spatial modelling.

## 9. Phase 4B closure requirements

Before Phase 4B is formally closed:

1. the complete automated test suite must pass;
2. staged changes must pass `git diff --cached --check`;
3. the generated reference-baseline artifact must be included;
4. the working tree must be clean after the closure commit.

Phase 4C must not begin until these checks pass.
