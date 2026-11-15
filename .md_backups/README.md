# HELIOS

### A modular analytics and machine learning workspace

HELIOS is a Python-based analytics application built with Streamlit. It brings data exploration, statistical analysis, regression modeling, scenario analysis, Monte Carlo simulation, and time-aware model validation into one interactive workspace.

The goal is to make analytical workflows easier to execute, inspect, and communicate—without requiring users to build every analysis from scratch.

---

## Features

| Module | Description |
|---|---|
| **Overview** | Inspect uploaded datasets and review their structure |
| **Statistics** | Explore descriptive statistics and data distributions |
| **Regression** | Train regression models and examine model performance |
| **Scenario Analysis** | Explore how changes in input variables affect model predictions |
| **Monte Carlo Simulation** | Evaluate prediction behavior across repeated randomized simulations |
| **Purged Cross-Validation** | Evaluate models using time intervals, purging overlapping samples, and applying an embargo period |

### Regression and diagnostics

The regression workspace provides a modeling pipeline and performance evaluation tools. It supports:

- Model fitting and prediction
- Training and holdout evaluation, when a holdout is configured
- Regression performance metrics
- Coefficient inspection, where supported by the selected model
- Diagnostic and error analysis

Metrics should be interpreted in the context of the target variable, dataset, and evaluation split. Training performance alone is not evidence of generalization.

### Scenario analysis

Scenario analysis uses a fitted model to examine predictions under modified input conditions. It is intended for exploratory what-if analysis, not causal inference.

### Monte Carlo simulation

Monte Carlo simulation evaluates model outputs across repeated randomized input scenarios. Results depend on the input distributions and assumptions used in the simulation.

### Purged cross-validation

HELIOS includes a time-aware validation workflow designed for observations whose target periods overlap.

The validation engine supports:

- Combinatorial selection of test groups
- Purging training observations whose time intervals overlap test intervals
- Applying an embargo period after test intervals
- Reporting fold-level R², RMSE, and MAE
- Reporting purged and embargoed sample counts

**Important:** This implementation reports combinatorial validation splits. It does not currently reconstruct standard CPCV backtest paths or calculate Probability of Backtest Overfitting (PBO).

---

## Architecture

HELIOS uses a modular structure that separates the application interface, analytical engines, and shared functionality.

```text
HELIOS/
│
├── app.py
│
├── core/
│   ├── data_loader.py
│   └── session.py
│
├── modules/
│   ├── overview/
│   │   └── interface.py
│   │
│   ├── statistics/
│   │   └── interface.py
│   │
│   ├── regression/
│   │   ├── engine.py
│   │   ├── interface.py
│   │   └── diagnostics.py
│   │
│   ├── scenario/
│   │   ├── engine.py
│   │   └── interface.py
│   │
│   ├── simulation/
│   │   ├── engine.py
│   │   └── interface.py
│   │
│   └── validation/
│       ├── engine.py
│       └── interface.py
│
├── data/
│
└── README.md
```

Each module has a distinct responsibility. The interface handles user interaction, while the engine contains the corresponding analytical logic.

---

## Getting started

### Requirements

- Python 3.10 or later
- pip
- A supported desktop or laptop environment

### 1. Clone the repository

```bash
git clone <repository-url>
cd HELIOS
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows**
```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

If a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies used by the project and its modules.

### 4. Launch HELIOS

```bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open that address in your browser.

---

## Basic workflow

1. Launch the application.
2. Import a supported CSV or Excel dataset.
3. Inspect the dataset using the Overview and Statistics modules.
4. Configure a regression model by selecting the target and predictors.
5. Review the model's evaluation results and diagnostics.
6. Explore alternative inputs using Scenario Analysis.
7. Run Monte Carlo simulations to examine output variability.
8. Where valid time-interval metadata is available, run Purged Cross-Validation.

---

## Preparing data for Purged Cross-Validation

Unlike ordinary random cross-validation, purged validation requires meaningful observation intervals.

The dataset must contain two columns representing the beginning and end of each observation's information or target interval.

For example:

| Column | Meaning |
|---|---|
| `start_time` | Beginning of the observation's information interval |
| `end_time` | End of the observation's information or target interval |
| `future_7d_revenue` | Example prediction target |

The selected interval columns must contain valid, parseable timestamps, and each end time must be greater than or equal to its corresponding start time.

### Example configuration

For the included synthetic seven-day forecasting dataset:

- Start column: `start_time`
- End column: `end_time`
- Number of groups: `5`
- Test groups: `2`
- Embargo: `7D`

The embargo should reflect the time dependence and information overlap in the problem. It should not be treated as a universally correct default.

---

## Limitations and methodological considerations

HELIOS is an analytical prototype. Its outputs depend on the quality of the uploaded data, feature selection, modeling assumptions, and validation configuration.

A few important considerations:

- Scenario analysis is not causal analysis.
- Monte Carlo results depend on the distributions and assumptions supplied.
- Random train/test splits may be inappropriate for time-dependent datasets.
- Purged validation requires correctly defined time intervals.
- Combinatorial validation results should not be interpreted as a complete PBO analysis.
- Synthetic datasets demonstrate functionality; they do not establish real-world predictive performance.

HELIOS's current documented capabilities concern tabular data analytics and machine learning. It should not be represented as a dedicated NISAR/SAR image-processing pipeline unless a corresponding SAR ingestion and processing workflow is implemented.

---

## Reproducibility

For meaningful comparisons, record:

- Dataset and preprocessing choices
- Target and predictor variables
- Model configuration
- Train/test split configuration
- Validation group and embargo settings
- Evaluation metrics

Use the same data and configuration when comparing model results.

---

## Project status

**Status:** Prototype / hackathon development

HELIOS brings several analytical workflows into a single modular application. Its emphasis is on accessible modeling, exploratory analysis, and transparent evaluation.

---

## Acknowledgments

Developed as part of the NASA Space Apps Challenge project work.


