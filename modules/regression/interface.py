import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from core.data_loader import get_numeric_columns
from core.session import (
    get_model_result,
    set_model_result,
)

from modules.regression.engine import (
    run_ols,
    calculate_sensitivity,
)

from modules.regression.diagnostics import (
    calculate_residual_diagnostics,
    calculate_influence_diagnostics,
    calculate_qq_data,
)


def render(df):

    st.subheader("Regression Modeling Workspace")

    st.caption(
        "Pipeline-based OLS with training-only preprocessing, "
        "holdout evaluation, sensitivity analysis, and diagnostics."
    )

    numeric_columns = get_numeric_columns(df)

    if len(numeric_columns) < 2:
        st.warning("At least two numeric columns are required.")
        return

    # -----------------------------------------
    # MODEL CONFIGURATION
    # -----------------------------------------

    with st.form("helios_regression_form"):

        left, right = st.columns(2)

        with left:

            target = st.selectbox(
                "Target variable (Y)",
                numeric_columns
            )

            available_features = [
                column
                for column in numeric_columns
                if column != target
            ]

            features = st.multiselect(
                "Predictor variables (X)",
                available_features,
                default=available_features[
                    :min(3, len(available_features))
                ]
            )

            include_intercept = st.checkbox(
                "Include intercept",
                value=True
            )

        with right:

            test_fraction = st.slider(
                "Holdout fraction",
                0.0,
                0.5,
                0.2,
                0.05
            )

            random_seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=2_147_483_647,
                value=42,
                step=1
            )

            scale_features = st.checkbox(
                "Standardize predictors",
                value=True,
                help="Scaling is fitted on training data only."
            )

        submitted = st.form_submit_button(
            "Fit OLS pipeline",
            type="primary",
            disabled=not features
        )

    # -----------------------------------------
    # FIT PIPELINE
    # -----------------------------------------

    if submitted:

        try:

            result = run_ols(
                df=df,
                target=target,
                features=features,
                test_fraction=test_fraction,
                random_seed=int(random_seed),
                include_intercept=include_intercept,
                scale_features=scale_features,
            )

            set_model_result(result)

            st.success("Regression pipeline fitted.")

        except Exception as exc:

            set_model_result(None)

            st.error(f"Model fitting failed: {exc}")

    result = get_model_result()

    if result is None:
        return

    st.divider()

    # -----------------------------------------
    # MODEL INFORMATION
    # -----------------------------------------

    st.markdown("### Model Information")

    st.write(f"**Target:** `{result['target']}`")

    st.write(
        f"**Predictors:** {', '.join(result['features'])}"
    )

    st.write(
        "**Pipeline:** Median Imputation → "
        + (
            "Standardization → "
            if result["scale_features"]
            else ""
        )
        + "Linear Regression"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Complete observations", f"{result['n_total']:,}")
    c2.metric("Training observations", f"{result['n_train']:,}")
    c3.metric("Excluded observations", f"{result['n_excluded']:,}")

    # -----------------------------------------
    # PERFORMANCE METRICS
    # -----------------------------------------

    def display_metrics(metrics, prefix):

        columns = st.columns(6)

        metric_names = [
            ("r2", "R²"),
            ("adjusted_r2", "Adjusted R²"),
            ("rmse", "RMSE"),
            ("mae", "MAE"),
            ("nrmse", "NRMSE"),
            ("mape", "MAPE (%)"),
        ]

        for column, (key, label) in zip(
            columns,
            metric_names
        ):

            value = metrics[key]

            column.metric(
                f"{prefix} {label}",
                f"{value:.4f}"
                if np.isfinite(value)
                else "N/A"
            )

    st.markdown("### Training Performance")

    display_metrics(
        result["train_metrics"],
        "Train"
    )

    if result["test_metrics"] is not None:

        st.markdown("### Holdout Performance")

        display_metrics(
            result["test_metrics"],
            "Test"
        )

    else:

        st.info(
            "No holdout sample was configured. "
            "Only training metrics are available."
        )

    st.caption(
        "NRMSE = RMSE divided by target range. "
        "MAPE excludes zero-valued targets and is undefined "
        "when all targets are zero."
    )

    # -----------------------------------------
    # COEFFICIENTS
    # -----------------------------------------

    st.markdown("### Coefficient Estimates")

    st.dataframe(
        result["coefficients"].style.format(
            lambda x: f"{x:.5g}"
            if isinstance(x, (int, float, np.number))
            else str(x)
        ),
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------
    # RESULT TABS
    # -----------------------------------------

    model_tab, sensitivity_tab, diagnostic_tab, influence_tab, summary_tab = st.tabs(
        [
            "Predictions",
            "Sensitivity",
            "Residual Diagnostics",
            "Influence",
            "Model Summary"
        ]
    )

    # -----------------------------------------
    # PREDICTIONS
    # -----------------------------------------

    with model_tab:

        st.markdown("#### Observed vs. Predicted")

        train_predictions = result["train_predictions_df"]

        st.scatter_chart(
            train_predictions,
            x="Observed",
            y="Predicted",
            use_container_width=True
        )

        st.markdown("#### Training Residuals")

        st.scatter_chart(
            train_predictions,
            x="Predicted",
            y="Residual",
            use_container_width=True
        )

        prediction_frames = [train_predictions]

        if result["test_predictions_df"] is not None:

            prediction_frames.append(
                result["test_predictions_df"]
            )

        prediction_export = pd.concat(
            prediction_frames,
            ignore_index=True
        )

        st.download_button(
            "Download Predictions CSV",
            data=prediction_export.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="helios_predictions.csv",
            mime="text/csv"
        )

    # -----------------------------------------
    # SENSITIVITY ANALYSIS
    # -----------------------------------------

    with sensitivity_tab:

        st.markdown("#### One-at-a-time Sensitivity")

        st.write(
            "Perturb each predictor independently and measure "
            "the resulting change in model predictions."
        )

        perturbation = st.slider(
            "Perturbation size (training standard deviations)",
            min_value=0.05,
            max_value=1.0,
            value=0.10,
            step=0.05
        )

        if st.button("Run sensitivity analysis"):

            try:

                sensitivity = calculate_sensitivity(
                    result,
                    perturbation=perturbation
                )

                st.session_state[
                    "helios_sensitivity"
                ] = sensitivity

            except Exception as exc:

                st.error(
                    f"Sensitivity analysis failed: {exc}"
                )

        sensitivity = st.session_state.get(
            "helios_sensitivity"
        )

        if sensitivity is not None:

            st.dataframe(
                sensitivity,
                use_container_width=True,
                hide_index=True
            )

            st.markdown(
                "#### Mean Absolute Prediction Change"
            )

            chart_data = sensitivity.set_index(
                "Feature"
            )[
                ["Mean Absolute Prediction Change"]
            ]

            st.bar_chart(chart_data)

            st.download_button(
                "Download Sensitivity Results",
                data=sensitivity.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="helios_sensitivity.csv",
                mime="text/csv"
            )

        st.caption(
            "This is a local, one-at-a-time perturbation analysis. "
            "It is not a causal effect estimate and does not account "
            "for correlations among predictors."
        )

    # -----------------------------------------
    # RESIDUAL DIAGNOSTICS
    # -----------------------------------------

    with diagnostic_tab:

        diagnostics = calculate_residual_diagnostics(
            result
        )

        d1, d2, d3 = st.columns(3)

        d1.metric(
            "Durbin-Watson",
            f"{diagnostics['durbin_watson']:.4f}"
        )

        d2.metric(
            "Jarque-Bera p-value",
            f"{diagnostics['jarque_bera_pvalue']:.4g}"
        )

        bp = diagnostics["breusch_pagan"]

        d3.metric(
            "Breusch-Pagan p-value",
            f"{bp['lm_pvalue']:.4g}"
            if bp is not None
            else "N/A"
        )

        st.markdown("#### Residuals vs. Fitted")

        st.scatter_chart(
            result["train_predictions_df"],
            x="Predicted",
            y="Residual",
            use_container_width=True
        )

        st.markdown("#### Residual Distribution")

        fig, ax = plt.subplots()

        ax.hist(
            result["train_residuals"],
            bins="auto",
            edgecolor="black"
        )

        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### Normal Q-Q Plot")

        qq = calculate_qq_data(result)

        fig, ax = plt.subplots()

        ax.scatter(
            qq["theoretical"],
            qq["ordered"],
            s=18
        )

        line_x = np.array([
            np.min(qq["theoretical"]),
            np.max(qq["theoretical"])
        ])

        ax.plot(
            line_x,
            qq["intercept"] + qq["slope"] * line_x
        )

        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Ordered Residuals")

        st.pyplot(fig)
        plt.close(fig)

    # -----------------------------------------
    # INFLUENCE
    # -----------------------------------------

    with influence_tab:

        influence = calculate_influence_diagnostics(
            result
        )

        observations = influence["observations"]

        st.write(
            f"Leverage threshold: "
            f"{influence['leverage_threshold']:.4f}"
        )

        st.write(
            f"Cook's distance threshold: "
            f"{influence['cooks_threshold']:.4f}"
        )

        st.dataframe(
            observations.sort_values(
                "Cook's Distance",
                ascending=False
            ).head(25),
            use_container_width=True,
            hide_index=True
        )

        st.scatter_chart(
            observations,
            x="Leverage",
            y="Cook's Distance",
            use_container_width=True
        )

    # -----------------------------------------
    # MODEL SUMMARY
    # -----------------------------------------

    with summary_tab:

        st.write(
            "The fitted estimator is a scikit-learn "
            "LinearRegression model inside a preprocessing pipeline."
        )

        st.write(
            f"Training R²: {result['train_metrics']['r2']:.4f}"
        )

        st.write(
            f"Pipeline steps: "
            f"{list(result['pipeline'].named_steps.keys())}"
        )

    st.info(
        "Regression metrics and sensitivity results describe "
        "model behavior under the selected data and specification. "
        "They do not establish causality."
    )
