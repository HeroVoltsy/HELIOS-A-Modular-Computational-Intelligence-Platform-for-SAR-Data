import hashlib

import streamlit as st

# ============================================================
# CORE IMPORTS
# ============================================================

from core.data_loader import load_dataset

from core.session import (
    initialize_session,
    set_dataset,
    get_dataset,
    get_dataset_name,
    get_model_result,
)

# ============================================================
# MODULE IMPORTS
# ============================================================

from modules.overview.interface import render as render_overview
from modules.statistics.interface import render as render_statistics
from modules.regression.interface import render as render_regression

from modules.scenario.interface import render_scenario
from modules.simulation.interface import render_simulation
from modules.validation.interface import render_validation


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HELIOS | Computational Intelligence Engine",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# APPLICATION INITIALIZATION
# ============================================================

initialize_session()

if "_dataset_signature" not in st.session_state:
    st.session_state["_dataset_signature"] = None


# ============================================================
# HEADER
# ============================================================

st.title("☀️ HELIOS")

st.caption(
    "Multidisciplinary Computational Intelligence & Decision Engine"
)

st.markdown(
    "**Engineering. Economics. Uncertainty. Decisions.**"
)

st.divider()


# ============================================================
# SIDEBAR — DATA MANAGEMENT
# ============================================================

with st.sidebar:

    st.title("HELIOS")

    st.caption("Computational Intelligence Platform")

    st.divider()

    st.subheader("Dataset")

    uploaded_file = st.file_uploader(
        "Import Dataset",
        type=["csv", "xlsx", "xls"],
        help="Upload a CSV or Excel dataset.",
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        file_signature = hashlib.sha256(
            file_bytes
        ).hexdigest()

        # Only reload when the actual uploaded content changes.
        if file_signature != st.session_state["_dataset_signature"]:

            try:
                uploaded_file.seek(0)

                df = load_dataset(uploaded_file)

                set_dataset(
                    df,
                    uploaded_file.name,
                )

                st.session_state["_dataset_signature"] = (
                    file_signature
                )

                st.success("Dataset loaded successfully.")

            except Exception as error:

                st.error(
                    f"Dataset loading failed: {error}"
                )

    current_dataset = get_dataset()

    if current_dataset is not None:

        st.success("Dataset available")

        st.caption(
            f"File: {get_dataset_name()}"
        )

        st.caption(
            f"Rows: {len(current_dataset):,}"
        )

        st.caption(
            f"Columns: {len(current_dataset.columns):,}"
        )

    else:

        st.info("Upload a dataset to begin.")

    st.divider()


# ============================================================
# NAVIGATION
# ============================================================

PAGES = [
    "Overview",
    "Statistics",
    "Regression",
    "Scenario Analysis",
    "Monte Carlo Simulation",
    "Purged Cross-Validation",
]

with st.sidebar:

    st.subheader("Navigation")

    selected_page = st.radio(
        "Select Module",
        PAGES,
        index=0,
        key="helios_navigation",
    )

    st.divider()

    st.caption("HELIOS | Development Build")


# ============================================================
# DATASET VALIDATION
# ============================================================

df = get_dataset()

if df is None:

    st.title("Welcome to HELIOS")

    st.markdown(
        """
        HELIOS is a modular computational intelligence
        and decision-support platform.

        Upload a dataset using the sidebar to begin.

        ### Available modules

        - **Overview:** Dataset structure and quality.
        - **Statistics:** Descriptive and exploratory analysis.
        - **Regression:** Predictive modeling and diagnostics.
        - **Scenario Analysis:** Evaluate alternative input conditions.
        - **Monte Carlo Simulation:** Quantify uncertainty through simulation.
        - **Purged Cross-Validation:** Evaluate models using interval-aware validation.
        """
    )

    st.stop()


# ============================================================
# MODULE ROUTING
# ============================================================

if selected_page == "Overview":

    render_overview(df)


elif selected_page == "Statistics":

    render_statistics(df)


elif selected_page == "Regression":

    render_regression(df)


elif selected_page == "Scenario Analysis":

    model_result = get_model_result()

    render_scenario(model_result)


elif selected_page == "Monte Carlo Simulation":

    model_result = get_model_result()

    render_simulation(model_result)


elif selected_page == "Purged Cross-Validation":

    model_result = get_model_result()

    render_validation(
        model_result,
        df,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "HELIOS — Multidisciplinary Computational Intelligence & Decision Engine"
)
