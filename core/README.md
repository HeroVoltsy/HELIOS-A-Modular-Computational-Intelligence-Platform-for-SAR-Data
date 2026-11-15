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

<!-- HELIOS-AUTO-DOCS:START -->
## HELIOS Implementation: Core Infrastructure

The core layer manages dataset ingestion and application state.

Components:
- Data loading.
- CSV and Excel ingestion.
- Centralized session management.
- Dataset and model state.
- Invalidation of dependent analytical results.
<!-- HELIOS-AUTO-DOCS:END -->
