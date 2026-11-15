
from pathlib import Path

DOCS = {
    "core/README.md": """
# Core

Shared application infrastructure used by HELIOS modules.

## Responsibilities

- Data loading and import handling.
- Shared session-state management.
- Common functionality required by the application.

## Components

| File | Responsibility |
|---|---|
| `data_loader.py` | Handles dataset loading and import operations. |
| `session.py` | Manages shared application session state. |

## Design

The core layer separates shared application functionality from individual analytical modules.

It does not perform SAR preprocessing. It provides infrastructure for loading and managing datasets used by the analytical components.
""",

    "modules/overview/README.md": """
# Overview Module

The Overview module provides an initial inspection interface for uploaded datasets.

## Responsibilities

- Present dataset information.
- Display basic structural information.
- Help users understand the data before analysis.

## Components

| File | Responsibility |
|---|---|
| `interface.py` | Implements the overview interface. |

## Role in HELIOS

The Overview module serves as the initial dataset inspection workspace.

It helps users assess the available data before proceeding to statistical analysis or modeling.
""",

    "modules/statistics/README.md": """
# Statistics Module

The Statistics module provides descriptive and exploratory statistical analysis.

## Responsibilities

- Summarize numerical variables.
- Inspect distributions and descriptive statistics.
- Support initial data exploration.

## Components

| File | Responsibility |
|---|---|
| `interface.py` | Implements the statistical analysis interface. |

## Role in HELIOS

This module helps users understand the characteristics and variability of their datasets before applying predictive models.

For SAR-derived tabular data, it can support exploratory analysis of measurements such as backscatter values and derived features, provided these are imported in a supported tabular format.
""",

    "modules/regression/README.md": """
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
""",

    "modules/scenario/README.md": """
# Scenario Analysis Module

The Scenario Analysis module supports exploratory what-if analysis using a fitted model.

## Responsibilities

- Modify selected predictor values.
- Evaluate model predictions under alternative input conditions.
- Compare scenario outputs.

## Components

| File | Responsibility |
|---|---|
| `engine.py` | Implements scenario evaluation logic. |
| `interface.py` | Provides the scenario analysis interface. |

## Workflow

1. Obtain a fitted regression model.
2. Select or modify input variables.
3. Generate predictions for alternative scenarios.
4. Compare the resulting outputs.

## Interpretation

Scenario analysis is intended for model-based exploration.

A change in predicted output does not establish a causal relationship. Results depend on the fitted model and the validity of the supplied inputs.

## SAR Applications

Potential applications include exploring how changes in SAR-derived measurements or geospatial features relate to model predictions.
""",

    "modules/simulation/README.md": """
# Monte Carlo Simulation Module

The Simulation module supports repeated randomized evaluation of model outputs.

## Responsibilities

- Generate repeated simulation inputs.
- Evaluate model outputs across simulated scenarios.
- Summarize variability in simulated predictions.

## Components

| File | Responsibility |
|---|---|
| `engine.py` | Implements simulation logic. |
| `interface.py` | Provides the simulation interface. |

## Workflow

1. Provide a fitted model and input data.
2. Configure simulation parameters.
3. Execute repeated simulations.
4. Inspect the resulting output distribution.

## Interpretation

Simulation results depend on the input distributions, parameter assumptions, and model behavior.

Monte Carlo simulation does not automatically quantify every source of uncertainty. Users should document the assumptions used.

## SAR Applications

The module can be used to explore prediction variability when SAR-derived measurements or other geospatial features are varied under specified assumptions.
""",

    "modules/validation/README.md": """
# Purged Cross-Validation Module

The Validation module provides a time-aware combinatorial validation workflow for datasets containing observation intervals.

## Responsibilities

- Generate combinatorial test-group selections.
- Identify overlapping training and test intervals.
- Purge overlapping training observations.
- Apply an embargo period.
- Report fold-level model evaluation metrics.

## Components

| File | Responsibility |
|---|---|
| `engine.py` | Implements the purged combinatorial validation logic. |
| `interface.py` | Provides validation configuration and result presentation. |

## Required Data

The validation workflow requires valid timestamp columns representing the beginning and end of each observation's information or target interval.

Example:

| Column | Description |
|---|---|
| `start_time` | Beginning of the observation interval. |
| `end_time` | End of the observation interval. |

Both columns must contain parseable timestamps, and each end time must be greater than or equal to its start time.

## Validation Workflow

1. Select the interval start and end columns.
2. Configure the number of groups.
3. Configure the number of test groups.
4. Set an embargo period.
5. Execute validation.
6. Inspect fold-level metrics and sample counts.

## Reported Metrics

- R²
- RMSE
- MAE
- Purged sample count
- Embargoed sample count

## Methodological Limitations

The current implementation evaluates combinatorial splits and reports their metrics. It does not reconstruct standard CPCV backtest paths or calculate Probability of Backtest Overfitting (PBO).

The validity of the results depends on correctly defined observation intervals and an appropriate modeling split.

## SAR Applications

This workflow may be useful for temporally dependent SAR-derived observations, particularly where target or information intervals overlap.
"""
}

def main():
    root = Path(__file__).resolve().parent

    for relative_path, content in DOCS.items():
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            content.strip() + "\n",
            encoding="utf-8"
        )
        print(f"Created: {relative_path}")

    print(f"\nGenerated {len(DOCS)} module README files.")

if __name__ == "__main__":
    main()
