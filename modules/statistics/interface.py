
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from scipy import stats


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

def calculate_descriptive_statistics(df):

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        return None

    result = numeric_df.describe().T

    result["variance"] = numeric_df.var()
    result["skewness"] = numeric_df.skew()
    result["kurtosis"] = numeric_df.kurtosis()

    return result


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def calculate_correlation(df, method="pearson"):

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        return None

    return numeric_df.corr(method=method)


# ============================================================
# NORMALITY TESTING
# ============================================================

def run_normality_test(series):

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna().to_numpy()

    if len(values) < 3:
        raise ValueError(
            "At least three valid observations are required."
        )

    if np.unique(values).size < 2:
        raise ValueError(
            "Normality testing requires non-constant data."
        )

    # Shapiro-Wilk is computationally expensive for
    # very large samples. Randomly sample at most 5000.
    if len(values) > 5000:
        rng = np.random.default_rng(42)
        values = rng.choice(values, 5000, replace=False)

    statistic, p_value = stats.shapiro(values)

    return {
        "test": "Shapiro-Wilk",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "sample_size": len(values),
        "interpretation": (
            "Evidence against normality"
            if p_value < 0.05
            else "Insufficient evidence against normality"
        ),
    }


# ============================================================
# PEARSON CORRELATION TEST
# ============================================================

def run_pearson_test(df, x_column, y_column):

    paired = df[[x_column, y_column]].apply(
        pd.to_numeric,
        errors="coerce"
    ).dropna()

    if len(paired) < 3:
        raise ValueError(
            "At least three paired observations are required."
        )

    if paired[x_column].nunique() < 2 or paired[y_column].nunique() < 2:
        raise ValueError(
            "Both variables must have non-zero variance."
        )

    result = stats.pearsonr(
        paired[x_column],
        paired[y_column]
    )

    return {
        "correlation": float(result.statistic),
        "p_value": float(result.pvalue),
        "sample_size": len(paired),
    }


# ============================================================
# INDEPENDENT T-TEST
# ============================================================

def run_independent_ttest(df, numeric_column, group_column):

    data = df[[numeric_column, group_column]].copy()

    data[numeric_column] = pd.to_numeric(
        data[numeric_column],
        errors="coerce"
    )

    data = data.dropna()

    groups = data[group_column].unique()

    if len(groups) != 2:
        raise ValueError(
            "The selected grouping variable must have exactly two groups."
        )

    group_a = data.loc[
        data[group_column] == groups[0],
        numeric_column
    ]

    group_b = data.loc[
        data[group_column] == groups[1],
        numeric_column
    ]

    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError(
            "Each group must contain at least two observations."
        )

    if group_a.nunique() < 2 or group_b.nunique() < 2:
        raise ValueError(
            "Each group must have non-zero variance."
        )

    result = stats.ttest_ind(
        group_a,
        group_b,
        equal_var=False
    )

    return {
        "group_a": str(groups[0]),
        "group_b": str(groups[1]),
        "mean_a": float(group_a.mean()),
        "mean_b": float(group_b.mean()),
        "difference": float(group_a.mean() - group_b.mean()),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_a": len(group_a),
        "n_b": len(group_b),
    }


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

def calculate_mean_confidence_interval(series, confidence=0.95):

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(values) < 2:
        raise ValueError(
            "At least two observations are required."
        )

    mean = values.mean()
    standard_error = stats.sem(values)

    lower, upper = stats.t.interval(
        confidence,
        df=len(values) - 1,
        loc=mean,
        scale=standard_error
    )

    return {
        "mean": float(mean),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": confidence,
        "sample_size": len(values),
    }


# ============================================================
# DESCRIPTIVE STATISTICS INTERFACE
# ============================================================

def render_descriptive(df):

    st.subheader("Descriptive Statistics")

    result = calculate_descriptive_statistics(df)

    if result is None:
        st.info("No numeric variables available.")
        return

    st.dataframe(
        result,
        use_container_width=True
    )

    st.download_button(
        "Download Descriptive Statistics",
        result.to_csv().encode("utf-8"),
        "helios_descriptive_statistics.csv",
        "text/csv"
    )


# ============================================================
# DISTRIBUTION INTERFACE
# ============================================================

def render_distribution(df):

    st.subheader("Distribution Analysis")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_columns:
        st.info("No numeric variables available.")
        return

    column = st.selectbox(
        "Select Variable",
        numeric_columns,
        key="statistics_distribution_column"
    )

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        st.warning("No valid observations.")
        return

    fig = px.histogram(
        x=values,
        nbins=40,
        marginal="box",
        title=f"Distribution of {column}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    try:
        normality = run_normality_test(values)

        st.write("**Shapiro-Wilk Normality Test**")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Test Statistic",
            f"{normality['statistic']:.4f}"
        )

        col2.metric(
            "P-value",
            f"{normality['p_value']:.6f}"
        )

        col3.metric(
            "Sample Size",
            normality["sample_size"]
        )

        st.caption(normality["interpretation"])

    except ValueError as error:
        st.warning(str(error))


# ============================================================
# CORRELATION INTERFACE
# ============================================================

def render_correlation(df):

    st.subheader("Correlation Analysis")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if len(numeric_columns) < 2:
        st.info("At least two numeric variables are required.")
        return

    method = st.selectbox(
        "Correlation Method",
        ["pearson", "spearman", "kendall"]
    )

    correlation = calculate_correlation(
        df,
        method
    )

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=f"{method.title()} Correlation Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# HYPOTHESIS TESTING INTERFACE
# ============================================================

def render_hypothesis_testing(df):

    st.subheader("Hypothesis Testing")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    test_type = st.selectbox(
        "Select Statistical Test",
        [
            "Pearson Correlation",
            "Independent Welch T-Test",
            "Mean Confidence Interval",
        ]
    )

    if test_type == "Pearson Correlation":

        if len(numeric_columns) < 2:
            st.info("At least two numeric variables are required.")
            return

        x_column = st.selectbox(
            "Variable X",
            numeric_columns,
            key="pearson_x"
        )

        y_candidates = [
            col for col in numeric_columns
            if col != x_column
        ]

        y_column = st.selectbox(
            "Variable Y",
            y_candidates,
            key="pearson_y"
        )

        if st.button("Run Pearson Test"):

            try:
                result = run_pearson_test(
                    df,
                    x_column,
                    y_column
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Correlation",
                    f"{result['correlation']:.4f}"
                )

                col2.metric(
                    "P-value",
                    f"{result['p_value']:.6f}"
                )

                col3.metric(
                    "Sample Size",
                    result["sample_size"]
                )

                if result["p_value"] < 0.05:
                    st.success(
                        "Reject the null hypothesis of zero population correlation."
                    )
                else:
                    st.info(
                        "Insufficient evidence to reject the null hypothesis."
                    )

            except ValueError as error:
                st.error(str(error))

    elif test_type == "Independent Welch T-Test":

        if not numeric_columns or not categorical_columns:
            st.info(
                "A numeric variable and categorical grouping variable are required."
            )
            return

        numeric_column = st.selectbox(
            "Numeric Variable",
            numeric_columns
        )

        group_column = st.selectbox(
            "Grouping Variable",
            categorical_columns
        )

        if st.button("Run Welch T-Test"):

            try:
                result = run_independent_ttest(
                    df,
                    numeric_column,
                    group_column
                )

                st.write(
                    f"Groups: {result['group_a']} vs {result['group_b']}"
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Mean Difference",
                    f"{result['difference']:.4f}"
                )

                col2.metric(
                    "T Statistic",
                    f"{result['statistic']:.4f}"
                )

                col3.metric(
                    "P-value",
                    f"{result['p_value']:.6f}"
                )

                if result["p_value"] < 0.05:
                    st.success(
                        "Reject the null hypothesis of equal population means."
                    )
                else:
                    st.info(
                        "Insufficient evidence to reject the null hypothesis."
                    )

            except ValueError as error:
                st.error(str(error))

    elif test_type == "Mean Confidence Interval":

        if not numeric_columns:
            st.info("No numeric variables available.")
            return

        column = st.selectbox(
            "Select Variable",
            numeric_columns
        )

        confidence = st.select_slider(
            "Confidence Level",
            options=[0.90, 0.95, 0.99],
            value=0.95
        )

        if st.button("Calculate Confidence Interval"):

            try:
                result = calculate_mean_confidence_interval(
                    df[column],
                    confidence
                )

                st.metric(
                    "Sample Mean",
                    f"{result['mean']:.4f}"
                )

                st.write(
                    f"{confidence:.0%} Confidence Interval: "
                    f"[{result['lower']:.4f}, {result['upper']:.4f}]"
                )

                st.caption(
                    f"Sample size: {result['sample_size']:,}"
                )

            except ValueError as error:
                st.error(str(error))


# ============================================================
# MAIN INTERFACE
# ============================================================

def render(df):

    st.title("Statistical Analysis")

    st.caption(
        "Descriptive statistics, distribution analysis, "
        "correlation, and hypothesis testing."
    )

    if df is None or df.empty:
        st.warning("No dataset is currently available.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "Descriptive Statistics",
        "Distribution Analysis",
        "Correlation",
        "Hypothesis Testing"
    ])

    with tab1:
        render_descriptive(df)

    with tab2:
        render_distribution(df)

    with tab3:
        render_correlation(df)

    with tab4:
        render_hypothesis_testing(df)
