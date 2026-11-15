# Regression Module

The Regression module provides the model-fitting and evaluation workspace in HELIOS.

## Responsibilities

- Configure regression targets and predictors.
- Fit regression models.
- Generate predictions.
- Evaluate model performance.
- Inspect regression diagnostics.

## Components

| File | Responsibility |
|---|---|
| `engine.py` | Contains the regression modeling logic. |
| `interface.py` | Provides the modeling interface. |
| `diagnostics.py` | Provides regression diagnostic functionality. |

## Modeling Workflow

1. Select the target variable.
2. Select predictor variables.
3. Configure the model.
4. Fit the modeling pipeline.
5. Examine predictions and evaluation metrics.
6. Inspect available diagnostics.

## Leakage Considerations

The modeling workflow uses a pipeline to help keep preprocessing operations within the model-fitting process.

Users should still verify that the selected predictors are available at prediction time and that the evaluation split reflects the data's temporal or spatial structure.

## SAR Applications

The module can analyze tabular SAR-derived features, including backscatter measurements, polarization features, and temporal observations.

It does not independently process raw SAR imagery.
