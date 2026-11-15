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

<!-- HELIOS-AUTO-DOCS:START -->
## HELIOS Implementation: Purged Cross-Validation

The Validation module implements a temporal Purged
Cross-Validation workflow.

It is designed to reduce leakage between overlapping
temporal observations.

Important: observation start_time and end_time must be
valid and correctly ordered. Temporal validation results
must be verified against the dataset before being reported.
<!-- HELIOS-AUTO-DOCS:END -->
