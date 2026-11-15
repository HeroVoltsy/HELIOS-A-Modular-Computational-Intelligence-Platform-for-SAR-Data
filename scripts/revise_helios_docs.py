
from pathlib import Path
import shutil

ROOT = Path(".").resolve()
BACKUP = ROOT / ".md_backups"

START = "<!-- HELIOS-AUTO-DOCS:START -->"
END = "<!-- HELIOS-AUTO-DOCS:END -->"

DOCUMENTATION = r"""
# HELIOS — Implementation Documentation

## 1. Project Overview

HELIOS is a modular analytical engine for preprocessed Synthetic Aperture Radar
(SAR) data and general structured datasets.

Built with Python and Streamlit, HELIOS integrates statistical analysis,
regression modeling, diagnostics, sensitivity analysis, scenario evaluation,
Monte Carlo simulation, and temporal validation into a unified application.

HELIOS is an analytical platform, not a raw SAR signal-processing system.

## 2. Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Statsmodels
- Plotly
- CSV and Excel data ingestion

## 3. Application Architecture

The application follows a modular architecture.

### Core

- `app.py`: Main application controller and navigation.
- `core/data_loader.py`: Dataset ingestion and loading.
- `core/session.py`: Centralized application state management.

### Analytical modules

- `modules/overview/`: Dataset inspection and profiling.
- `modules/statistics/`: Descriptive statistical analysis.
- `modules/regression/`: Regression modeling and diagnostics.
- `modules/scenario/`: Scenario evaluation.
- `modules/simulation/`: Monte Carlo simulation.
- `modules/validation/`: Purged Cross-Validation.
- Feature engineering components: Feature transformations and expansion.

Each module separates its analytical engine from its Streamlit interface.

## 4. Data Management and Inspection

HELIOS supports CSV and Excel dataset ingestion.

The application provides:

- Dataset preview and inspection.
- Column metadata and data-type identification.
- Missing-value analysis.
- Duplicate observation detection.
- Constant-column detection.
- Numeric and categorical summaries.
- Dataset profiling.
- Data export.

Exploratory inspection does not automatically modify the
original dataset or silently remove observations.

## 5. Statistical Analysis

The statistics workspace provides:

- Descriptive statistics.
- Distribution analysis.
- Pearson correlation analysis.
- Numeric variable summaries.
- Categorical variable summaries.
- Missingness and data-quality inspection.
- Statistical visualizations.

## 6. Regression Modeling Engine

The regression engine uses Scikit-learn and a Pipeline-based
modeling workflow.

Implemented modeling components include:

- Train/test partitioning.
- Median imputation.
- Optional feature scaling.
- Linear regression.
- Prediction generation.
- Training and testing evaluation.
- Model performance reporting.

Preprocessing is fitted within the training pipeline to reduce
data leakage during model evaluation.

### Regression metrics

The evaluation framework includes:

- R-squared.
- Adjusted R-squared.
- Root Mean Squared Error (RMSE).
- Mean Absolute Error (MAE).
- Normalized RMSE (NRMSE).
- Mean Absolute Percentage Error (MAPE).
- Prediction error analysis.

## 7. Regression Diagnostics

The diagnostics workspace evaluates model assumptions
and regression behavior.

Implemented diagnostic procedures include:

- Residual analysis.
- Residual visualization.
- Durbin-Watson test.
- Jarque-Bera test.
- Breusch-Pagan test.
- Leverage analysis.

These procedures provide additional information beyond
predictive performance metrics.

Diagnostic results should be interpreted in the context
of the assumptions of each statistical test.

## 8. Sensitivity Analysis

HELIOS includes feature perturbation-based sensitivity analysis.

The objective is to evaluate how changes in input variables
affect model predictions.

This supports:

- Input sensitivity evaluation.
- Feature-level response analysis.
- Model behavior inspection.
- Comparison of prediction changes.

Sensitivity results are model-dependent and should not
automatically be interpreted as causal effects.

## 9. Feature Engineering

The feature engineering components support transformations
of analytical input variables.

Implemented transformation work includes:

- Standardization.
- Min-max scaling.
- Polynomial feature expansion.
- Constant-feature validation.

Transformations must be fitted using training data only
when used in predictive modeling workflows.

## 10. Scenario Analysis

The scenario analysis module evaluates alternative input
conditions using an existing analytical model.

It supports individual and multiple-scenario evaluation.

The purpose is to examine how different assumptions
and input configurations influence model outputs.

Scenario results are conditional model outputs,
not guaranteed real-world outcomes.

## 11. Monte Carlo Simulation

The simulation module provides a stochastic analytical
workspace for examining distributions of possible outcomes.

It is intended to support:

- Repeated simulation.
- Outcome distribution analysis.
- Uncertainty exploration.
- Comparison of simulated outcomes.

Simulation assumptions and input distributions must be
documented when interpreting results.

## 12. Purged Cross-Validation

The validation module implements a Purged Cross-Validation
workflow intended for temporally dependent observations.

Its purpose is to reduce information leakage between
training and validation observations.

The workflow requires valid observation start and end times
and appropriate temporal interval handling.

CPCV implementation and edge cases must be verified
against the actual dataset before treating results
as validated experimental evidence.

## 13. SAR Analytical Workflow

HELIOS has been extended to support analysis of preprocessed
Synthetic Aperture Radar datasets.

The project includes a sample NISAR-derived analytical dataset.

The SAR workflow focuses on:

- Preprocessed SAR observation analysis.
- Backscatter measurements.
- Polarization-specific variables.
- Statistical characterization.
- Regression and analytical modeling.
- Sensitivity and validation workflows.

### Sample dataset

The prepared `frequencyA_helios.csv` dataset contains:

- 53,055 observations.
- HHHH backscatter measurements.
- HVHV backscatter measurements.
- Validated decibel representations of backscatter.

The dataset preparation workflow is implemented separately
from the Streamlit analytical interface.

HELIOS does not claim to perform the complete raw SAR
processing chain, including raw acquisition decoding,
focusing, calibration, geocoding, or interferometric
processing.

## 14. Session and State Management

The application maintains centralized session state for:

- Loaded datasets.
- Regression models and results.
- Scenario analysis results.
- Multiple-scenario results.
- Monte Carlo simulation results.
- Validation results.

Dataset changes invalidate dependent analytical results.
Model changes invalidate downstream results that depend
on the previous model.

This reduces the possibility of displaying stale outputs.

## 15. Design Principles

HELIOS follows these development principles:

1. Modular analytical components.
2. Reproducible modeling workflows.
3. Explicit validation procedures.
4. Leakage-conscious preprocessing.
5. Separation of computation and presentation.
6. Transparent statistical outputs.
7. Practical analytical functionality over unnecessary complexity.

## 16. Scope and Limitations

HELIOS is designed as an analytical engine.

Its capabilities should not be confused with:

- A complete raw SAR processing suite.
- A replacement for specialized SAR calibration software.
- A causal inference engine.
- An automatic scientific interpretation system.
- A guarantee of predictive accuracy.

Individual analytical modules require dataset-specific
validation and appropriate statistical assumptions.

## 17. Development Status

The application has progressed from an initial
Streamlit-based statistical analysis MVP into a
multi-module analytical environment.

Its development has expanded across data inspection,
regression, diagnostics, sensitivity analysis,
scenario evaluation, simulation, temporal validation,
feature engineering, and preprocessed SAR analysis.

Implementation status should be assessed against the
current source code and successful runtime tests.
"""

MODULE_DOCS = {
    "overview": """
## HELIOS Implementation: Overview

The Overview module provides dataset-level inspection and profiling.

Implemented capabilities:
- Dataset preview and metadata.
- Column type identification.
- Missingness analysis.
- Duplicate and constant-column checks.
- Numeric and categorical summaries.
- Distribution visualization.
- Pearson correlation.
- Dataset export and inspection.

The module is descriptive. It does not automatically
clean or mutate the original dataset.
""",
    "statistics": """
## HELIOS Implementation: Statistics

The Statistics module provides descriptive statistical analysis,
distribution exploration, numeric and categorical summaries,
and correlation analysis.

It is intended for exploratory data analysis rather than
automatic statistical inference.
""",
    "regression": """
## HELIOS Implementation: Regression

The Regression module provides a Pipeline-based modeling workflow.

Implemented capabilities:
- Train/test splitting.
- Median imputation.
- Optional scaling.
- Linear regression.
- Prediction and evaluation.
- R-squared and adjusted R-squared.
- RMSE, MAE, NRMSE, and MAPE.
- Sensitivity analysis.
- Residual diagnostics.
- Durbin-Watson, Jarque-Bera, and Breusch-Pagan tests.
- Leverage analysis.

Preprocessing is performed within the modeling pipeline
to reduce leakage from fitting transformations before
the training/test partition.
""",
    "scenario": """
## HELIOS Implementation: Scenario Analysis

The Scenario module evaluates alternative input configurations
using an existing model.

It supports individual and multiple-scenario analysis
to compare conditional model outputs.
""",
    "simulation": """
## HELIOS Implementation: Monte Carlo Simulation

The Simulation module provides repeated stochastic evaluation
and outcome distribution analysis.

Its purpose is uncertainty exploration under explicitly
specified simulation assumptions.
""",
    "validation": """
## HELIOS Implementation: Purged Cross-Validation

The Validation module implements a temporal Purged
Cross-Validation workflow.

It is designed to reduce leakage between overlapping
temporal observations.

Important: observation start_time and end_time must be
valid and correctly ordered. Temporal validation results
must be verified against the dataset before being reported.
""",
    "feature_engineering": """
## HELIOS Implementation: Feature Engineering

Feature engineering work includes:
- Standardization.
- Min-max scaling.
- Polynomial expansion.
- Constant-feature validation.

Predictive preprocessing should be fitted exclusively
on training observations.
""",
    "core": """
## HELIOS Implementation: Core Infrastructure

The core layer manages dataset ingestion and application state.

Components:
- Data loading.
- CSV and Excel ingestion.
- Centralized session management.
- Dataset and model state.
- Invalidation of dependent analytical results.
"""
}


def replace_managed_section(content, section):
    block = f"{START}\n{section.strip()}\n{END}"

    if START in content and END in content:
        before = content.split(START)[0].rstrip()
        after = content.split(END, 1)[1].lstrip()
        return f"{before}\n\n{block}\n\n{after}".strip() + "\n"

    return content.rstrip() + "\n\n" + block + "\n"


def backup_file(path):
    relative = path.relative_to(ROOT)
    destination = BACKUP / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def write_document(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        old = path.read_text(encoding="utf-8")
        if old == content:
            return False
        backup_file(path)

    path.write_text(content, encoding="utf-8")
    print(f"UPDATED: {path.relative_to(ROOT)}")
    return True


def classify_module(path):
    parts = {part.lower() for part in path.parts}

    for module in MODULE_DOCS:
        if module in parts:
            return module

    return None


def main():
    # 1. Create comprehensive implementation documentation.
    docs_path = ROOT / "docs" / "IMPLEMENTATION.md"
    write_document(docs_path, DOCUMENTATION.strip() + "\n")

    # 2. Update the root README.
    root_readme = ROOT / "README.md"

    if root_readme.exists():
        content = root_readme.read_text(encoding="utf-8")
    else:
        content = "# HELIOS\n\n"

    root_section = """
## Implementation Documentation

HELIOS is a modular analytical engine for structured datasets
and preprocessed SAR data.

For the complete implementation history, architecture,
analytical modules, capabilities, and scope, see:

- [Implementation Documentation](docs/IMPLEMENTATION.md)
"""
    updated = replace_managed_section(content, root_section)
    write_document(root_readme, updated)

    # 3. Update existing module README files.
    readmes = list(ROOT.rglob("README.md"))

    excluded = {".git", ".venv", ".md_backups", "__pycache__"}

    for path in readmes:
        if any(part in excluded for part in path.parts):
            continue

        module = classify_module(path)

        if module is None:
            continue

        content = path.read_text(encoding="utf-8")
        updated = replace_managed_section(
            content,
            MODULE_DOCS[module]
        )
        write_document(path, updated)

    # 4. Print summary.
    print("\nHELIOS documentation revision complete.")
    print(f"Master documentation: {docs_path.relative_to(ROOT)}")
    print(f"Root README: {root_readme.relative_to(ROOT)}")
    print(f"Backups: {BACKUP.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
