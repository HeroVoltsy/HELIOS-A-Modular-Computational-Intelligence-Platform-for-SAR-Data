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

<!-- HELIOS-AUTO-DOCS:START -->
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
<!-- HELIOS-AUTO-DOCS:END -->
