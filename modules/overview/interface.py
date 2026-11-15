
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# ============================================================
# DATA PROFILING
# ============================================================

def profile_dataset(df):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    datetime_columns = df.select_dtypes(
        include=["datetime"]
    ).columns.tolist()

    missing_cells = int(df.isna().sum().sum())
    total_cells = int(df.size)

    missing_percentage = (
        missing_cells / total_cells * 100
        if total_cells > 0 else 0
    )

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "missing_cells": missing_cells,
        "missing_percentage": missing_percentage,
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage": float(
            df.memory_usage(deep=True).sum() / (1024 ** 2)
        ),
    }


# ============================================================
# COLUMN PROFILE
# ============================================================

def generate_column_profile(df):

    records = []

    for column in df.columns:

        series = df[column]

        records.append({
            "Column": column,
            "Data Type": str(series.dtype),
            "Non-Null": int(series.notna().sum()),
            "Missing": int(series.isna().sum()),
            "Missing (%)": round(
                series.isna().mean() * 100, 2
            ),
            "Unique Values": int(series.nunique()),
            "Sample Value": (
                str(series.dropna().iloc[0])
                if series.notna().any()
                else "No valid values"
            ),
        })

    return pd.DataFrame(records)


# ============================================================
# MISSINGNESS ANALYSIS
# ============================================================

def generate_missingness(df):

    missing = df.isna().sum()

    missing = missing[missing > 0]

    if missing.empty:
        return None

    result = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values,
        "Missing (%)": (
            missing.values / len(df) * 100
        ),
    })

    return result.sort_values(
        "Missing (%)",
        ascending=False,
    )


# ============================================================
# NUMERIC SUMMARY
# ============================================================

def generate_numeric_summary(df):

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.empty:
        return None

    return numeric_df.describe().T


# ============================================================
# CATEGORICAL SUMMARY
# ============================================================

def generate_categorical_summary(df):

    categorical_df = df.select_dtypes(
        include=["object", "category", "bool"]
    )

    if categorical_df.empty:
        return None

    records = []

    for column in categorical_df.columns:

        series = categorical_df[column]

        records.append({
            "Column": column,
            "Unique Values": int(series.nunique()),
            "Most Frequent": (
                str(series.mode().iloc[0])
                if not series.mode().empty
                else None
            ),
            "Frequency": (
                int(series.value_counts().iloc[0])
                if not series.value_counts().empty
                else 0
            ),
        })

    return pd.DataFrame(records)


# ============================================================
# DATASET OVERVIEW
# ============================================================

def render_metrics(profile):

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Observations",
        f"{profile['rows']:,}",
    )

    col2.metric(
        "Features",
        f"{profile['columns']:,}",
    )

    col3.metric(
        "Missing Cells",
        f"{profile['missing_cells']:,}",
    )

    col4.metric(
        "Duplicate Rows",
        f"{profile['duplicate_rows']:,}",
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Numeric Features",
        len(profile["numeric_columns"]),
    )

    col2.metric(
        "Categorical Features",
        len(profile["categorical_columns"]),
    )

    col3.metric(
        "Memory Usage",
        f"{profile['memory_usage']:.2f} MB",
    )


# ============================================================
# DATA QUALITY VISUALIZATION
# ============================================================

def render_missingness(missingness):

    st.subheader("Missing Data Analysis")

    if missingness is None:

        st.success(
            "No missing values detected in the dataset."
        )

        return

    fig = px.bar(
        missingness,
        x="Column",
        y="Missing (%)",
        color="Missing (%)",
        title="Missing Values by Feature",
        hover_data=["Missing Values"],
    )

    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="Missing Values (%)",
        xaxis_tickangle=-45,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.dataframe(
        missingness,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FEATURE DISTRIBUTION
# ============================================================

def render_feature_distribution(df, numeric_columns):

    st.subheader("Feature Distribution")

    if not numeric_columns:

        st.info(
            "No numeric columns available for distribution analysis."
        )

        return

    selected_column = st.selectbox(
        "Select Feature",
        numeric_columns,
        key="overview_distribution_column",
    )

    values = pd.to_numeric(
        df[selected_column],
        errors="coerce",
    ).dropna()

    if values.empty:

        st.warning(
            "This feature has no valid numeric observations."
        )

        return

    fig = px.histogram(
        x=values,
        nbins=40,
        title=f"Distribution of {selected_column}",
        labels={"x": selected_column},
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Mean", f"{values.mean():.4f}")
    col2.metric("Median", f"{values.median():.4f}")
    col3.metric("Std. Dev.", f"{values.std():.4f}")
    col4.metric("Min / Max", f"{values.min():.4f} / {values.max():.4f}")


# ============================================================
# CORRELATION OVERVIEW
# ============================================================

def render_correlation(df, numeric_columns):

    st.subheader("Feature Correlation")

    if len(numeric_columns) < 2:

        st.info(
            "At least two numeric columns are required."
        )

        return

    correlation = df[numeric_columns].corr(
        method="pearson"
    )

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pearson Correlation Matrix",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# DATA PREVIEW
# ============================================================

def render_data_preview(df):

    st.subheader("Dataset Preview")

    col1, col2 = st.columns([1, 1])

    with col1:

        rows_to_display = st.selectbox(
            "Preview Rows",
            [5, 10, 20, 50, 100],
            index=1,
        )

    with col2:

        show_tail = st.checkbox(
            "Show last rows",
            value=False,
        )

    preview = (
        df.tail(rows_to_display)
        if show_tail
        else df.head(rows_to_display)
    )

    st.dataframe(
        preview,
        use_container_width=True,
        hide_index=False,
    )


# ============================================================
# MAIN INTERFACE
# ============================================================

def render(df):

    st.title("Dataset Overview")

    st.caption(
        "Dataset profiling, structural inspection, "
        "and data quality assessment."
    )

    if df is None or df.empty:

        st.warning(
            "No dataset is currently available."
        )

        return

    profile = profile_dataset(df)

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    st.header("Dataset Summary")

    render_metrics(profile)

    st.divider()

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Preview",
        "Column Profile",
        "Data Quality",
        "Distributions",
        "Correlation",
    ])

    # --------------------------------------------------------
    # TAB 1: PREVIEW
    # --------------------------------------------------------

    with tab1:

        render_data_preview(df)

    # --------------------------------------------------------
    # TAB 2: COLUMN PROFILE
    # --------------------------------------------------------

    with tab2:

        st.subheader("Column-Level Profile")

        column_profile = generate_column_profile(df)

        st.dataframe(
            column_profile,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Column Profile",
            data=column_profile.to_csv(index=False),
            file_name="helios_column_profile.csv",
            mime="text/csv",
        )

        numeric_summary = generate_numeric_summary(df)

        if numeric_summary is not None:

            st.subheader("Numeric Summary Statistics")

            st.dataframe(
                numeric_summary,
                use_container_width=True,
            )

        categorical_summary = generate_categorical_summary(df)

        if categorical_summary is not None:

            st.subheader("Categorical Summary Statistics")

            st.dataframe(
                categorical_summary,
                use_container_width=True,
                hide_index=True,
            )

    # --------------------------------------------------------
    # TAB 3: DATA QUALITY
    # --------------------------------------------------------

    with tab3:

        st.subheader("Data Quality Assessment")

        missingness = generate_missingness(df)

        render_missingness(missingness)

        st.divider()

        duplicate_count = int(df.duplicated().sum())

        if duplicate_count > 0:

            st.warning(
                f"{duplicate_count:,} duplicate rows detected."
            )

        else:

            st.success(
                "No duplicate rows detected."
            )

        constant_columns = [
            column
            for column in df.columns
            if df[column].nunique(dropna=False) <= 1
        ]

        if constant_columns:

            st.warning(
                f"Constant or single-value columns detected: "
                f"{', '.join(map(str, constant_columns))}"
            )

        else:

            st.success(
                "No constant columns detected."
            )

    # --------------------------------------------------------
    # TAB 4: DISTRIBUTIONS
    # --------------------------------------------------------

    with tab4:

        render_feature_distribution(
            df,
            profile["numeric_columns"],
        )

    # --------------------------------------------------------
    # TAB 5: CORRELATION
    # --------------------------------------------------------

    with tab5:

        render_correlation(
            df,
            profile["numeric_columns"],
        )
