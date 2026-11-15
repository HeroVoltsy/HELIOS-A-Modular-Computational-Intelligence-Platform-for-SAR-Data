
import streamlit as st


# ============================================================
# SESSION DEFAULTS
# ============================================================

SESSION_DEFAULTS = {
    # Dataset
    "dataset": None,
    "dataset_name": None,

    # Regression
    "model_result": None,

    # Scenario Analysis
    "helios_scenario_result": None,
    "helios_multiple_scenario_results": None,

    # Monte Carlo Simulation
    "helios_simulation_result": None,

    # Purged Cross-Validation
    "helios_validation_result": None,
}


# ============================================================
# SESSION INITIALIZATION
# ============================================================

def initialize_session():
    """Initialize HELIOS session state without overwriting existing values."""

    for key, default_value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# ============================================================
# DATASET MANAGEMENT
# ============================================================

def set_dataset(df, filename):
    """Store a dataset and invalidate results from previous datasets."""

    st.session_state["dataset"] = df
    st.session_state["dataset_name"] = filename

    clear_analysis_results()


def get_dataset():
    """Retrieve the currently loaded dataset."""

    return st.session_state.get("dataset")


def get_dataset_name():
    """Retrieve the uploaded dataset filename."""

    return st.session_state.get("dataset_name")


def clear_dataset():
    """Remove the active dataset and all dependent analysis results."""

    st.session_state["dataset"] = None
    st.session_state["dataset_name"] = None

    clear_analysis_results()


# ============================================================
# MODEL MANAGEMENT
# ============================================================

def set_model_result(result):
    """Store a fitted regression result and invalidate downstream analyses."""

    st.session_state["model_result"] = result

    clear_downstream_results()


def get_model_result():
    """Retrieve the fitted regression result."""

    return st.session_state.get("model_result")


# ============================================================
# DOWNSTREAM RESULT MANAGEMENT
# ============================================================

def clear_downstream_results():
    """Clear results that depend on the fitted regression model."""

    st.session_state["helios_scenario_result"] = None
    st.session_state["helios_multiple_scenario_results"] = None
    st.session_state["helios_simulation_result"] = None
    st.session_state["helios_validation_result"] = None


def clear_analysis_results():
    """Clear all model and analysis results."""

    st.session_state["model_result"] = None

    clear_downstream_results()


# ============================================================
# SCENARIO ANALYSIS
# ============================================================

def set_scenario_result(result):
    st.session_state["helios_scenario_result"] = result


def get_scenario_result():
    return st.session_state.get("helios_scenario_result")


def set_multiple_scenario_results(result):
    st.session_state["helios_multiple_scenario_results"] = result


def get_multiple_scenario_results():
    return st.session_state.get("helios_multiple_scenario_results")


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================

def set_simulation_result(result):
    st.session_state["helios_simulation_result"] = result


def get_simulation_result():
    return st.session_state.get("helios_simulation_result")


# ============================================================
# PURGED CROSS-VALIDATION
# ============================================================

def set_validation_result(result):
    st.session_state["helios_validation_result"] = result


def get_validation_result():
    return st.session_state.get("helios_validation_result")
