
import streamlit as st
import pandas as pd

from modules.scenario.engine import (
    get_baseline_values,
    compare_scenarios,
    compare_multiple_scenarios,
)


def render_scenario(result):
    st.title("Scenario Analysis")

    st.caption(
        "Evaluate alternative predictor values using your fitted "
        "regression model."
    )

    if result is None:
        st.warning("Train a regression model before running scenario analysis.")
        return

    try:
        default_values = get_baseline_values(result)
    except (ValueError, KeyError) as error:
        st.error(str(error))
        return

    features = result["features"]

    st.subheader("Scenario Configuration")

    st.info(
        "Baseline values default to training-set medians. "
        "Adjust them if you want a different reference point."
    )

    baseline_values = {}
    scenario_values = {}

    col_baseline, col_scenario = st.columns(2)

    with col_baseline:
        st.markdown("#### Baseline")

        for feature in features:
            baseline_values[feature] = st.number_input(
                label=feature,
                value=float(default_values[feature]),
                key=f"scenario_baseline_{feature}",
                format="%.6f",
            )

    with col_scenario:
        st.markdown("#### Alternative Scenario")

        for feature in features:
            scenario_values[feature] = st.number_input(
                label=feature,
                value=float(default_values[feature]),
                key=f"scenario_alternative_{feature}",
                format="%.6f",
            )

    st.divider()

    if st.button(
        "Evaluate Scenario",
        type="primary",
        use_container_width=True,
    ):
        try:
            comparison = compare_scenarios(
                result=result,
                baseline_values=baseline_values,
                scenario_values=scenario_values,
            )

            st.session_state["helios_scenario_result"] = comparison

        except (ValueError, KeyError, TypeError) as error:
            st.error(f"Scenario evaluation failed: {error}")
            st.session_state.pop("helios_scenario_result", None)

    comparison = st.session_state.get(
        "helios_scenario_result"
    )

    if comparison is None:
        return

    st.subheader("Prediction Comparison")

    baseline_prediction = comparison["baseline_prediction"]
    scenario_prediction = comparison["scenario_prediction"]
    difference = comparison["absolute_difference"]
    percentage_change = comparison["percentage_change"]

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Baseline Prediction",
        f"{baseline_prediction:,.4f}",
    )

    metric2.metric(
        "Scenario Prediction",
        f"{scenario_prediction:,.4f}",
    )

    metric3.metric(
        "Absolute Difference",
        f"{difference:+,.4f}",
        delta=(
            f"{percentage_change:+.2f}%"
            if percentage_change is not None
            else None
        ),
    )

    if percentage_change is None:
        st.caption(
            "Percentage change is undefined because the baseline "
            "prediction is approximately zero."
        )

    st.subheader("Predictor Changes")

    feature_table = comparison["feature_comparison"].copy()

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True,
    )

    prediction_table = pd.DataFrame({
        "scenario": ["Baseline", "Alternative"],
        "predicted_value": [
            baseline_prediction,
            scenario_prediction,
        ],
    })

    st.subheader("Prediction Visualization")

    st.bar_chart(
        prediction_table.set_index("scenario"),
        y="predicted_value",
    )

    csv = feature_table.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Predictor Comparison",
        data=csv,
        file_name="helios_scenario_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()

    st.subheader("Multiple Scenario Comparison")

    st.caption(
        "Compare optimistic, conservative, or custom alternatives "
        "against the same baseline."
    )

    scenario_count = st.number_input(
        "Number of additional scenarios",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        key="helios_scenario_count",
    )

    multiple_scenarios = {}

    for index in range(int(scenario_count)):
        scenario_name = f"Scenario {index + 1}"

        with st.expander(scenario_name):
            values = {}

            for feature in features:
                values[feature] = st.number_input(
                    feature,
                    value=float(default_values[feature]),
                    key=f"helios_multi_{index}_{feature}",
                    format="%.6f",
                )

            multiple_scenarios[scenario_name] = values

    if st.button(
        "Compare All Scenarios",
        use_container_width=True,
    ):
        try:
            results = compare_multiple_scenarios(
                result=result,
                baseline_values=baseline_values,
                scenarios=multiple_scenarios,
            )

            st.session_state[
                "helios_multiple_scenario_results"
            ] = results

        except (ValueError, KeyError, TypeError) as error:
            st.error(f"Comparison failed: {error}")

    multiple_results = st.session_state.get(
        "helios_multiple_scenario_results"
    )

    if multiple_results is not None:
        st.dataframe(
            multiple_results,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download All Scenario Results",
            data=multiple_results.to_csv(index=False).encode("utf-8"),
            file_name="helios_multiple_scenarios.csv",
            mime="text/csv",
        )

    st.warning(
        "Scenario predictions describe model outputs under changed "
        "inputs. They do not establish causal effects or guarantee "
        "real-world outcomes."
    )
