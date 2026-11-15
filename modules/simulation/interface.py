
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from modules.simulation.engine import run_monte_carlo


def render_simulation(result):

    st.title("Monte Carlo Simulation")

    st.caption(
        "Evaluate prediction uncertainty using simulated "
        "input distributions."
    )

    if result is None:
        st.warning("Train a regression model first.")
        return

    features = result["features"]
    X_train = result["X_train"]

    st.subheader("Simulation Configuration")

    st.info(
        "Distribution parameters are configurable assumptions. "
        "They are not automatically estimated probability models."
    )

    distributions = {}

    for feature in features:

        with st.expander(feature, expanded=True):

            distribution = st.selectbox(
                f"Distribution for {feature}",
                ["normal", "uniform", "triangular"],
                key=f"sim_distribution_{feature}",
            )

            if distribution == "normal":

                mean = st.number_input(
                    "Mean",
                    value=float(X_train[feature].median()),
                    key=f"sim_mean_{feature}",
                )

                std = st.number_input(
                    "Standard deviation",
                    min_value=0.000001,
                    value=max(
                        float(X_train[feature].std()),
                        0.000001,
                    ),
                    key=f"sim_std_{feature}",
                )

                distributions[feature] = {
                    "distribution": "normal",
                    "mean": mean,
                    "std": std,
                }

            elif distribution == "uniform":

                minimum = float(X_train[feature].min())
                maximum = float(X_train[feature].max())

                low = st.number_input(
                    "Minimum",
                    value=minimum,
                    key=f"sim_low_{feature}",
                )

                high = st.number_input(
                    "Maximum",
                    value=maximum,
                    key=f"sim_high_{feature}",
                )

                distributions[feature] = {
                    "distribution": "uniform",
                    "low": low,
                    "high": high,
                }

            elif distribution == "triangular":

                minimum = float(X_train[feature].min())
                maximum = float(X_train[feature].max())
                mode = float(X_train[feature].median())

                low = st.number_input(
                    "Minimum",
                    value=minimum,
                    key=f"sim_tri_low_{feature}",
                )

                most_likely = st.number_input(
                    "Most likely value",
                    value=mode,
                    key=f"sim_tri_mode_{feature}",
                )

                high = st.number_input(
                    "Maximum",
                    value=maximum,
                    key=f"sim_tri_high_{feature}",
                )

                distributions[feature] = {
                    "distribution": "triangular",
                    "low": low,
                    "mode": most_likely,
                    "high": high,
                }

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        n_simulations = st.number_input(
            "Number of simulations",
            min_value=100,
            max_value=100000,
            value=10000,
            step=1000,
        )

    with col2:
        random_seed = st.number_input(
            "Random seed",
            min_value=0,
            value=42,
            step=1,
        )

    with col3:
        threshold = st.number_input(
            "Target threshold",
            value=0.0,
            help="Calculate the probability of exceeding this value.",
        )

    if st.button(
        "Run Monte Carlo Simulation",
        type="primary",
        use_container_width=True,
    ):

        try:

            simulation = run_monte_carlo(
                result=result,
                distributions=distributions,
                n_simulations=int(n_simulations),
                random_seed=int(random_seed),
                threshold=float(threshold),
            )

            st.session_state[
                "helios_simulation_result"
            ] = simulation

        except (ValueError, KeyError, TypeError) as error:

            st.error(f"Simulation failed: {error}")
            st.session_state["helios_simulation_result"] = None

    simulation = st.session_state.get(
        "helios_simulation_result"
    )

    if simulation is None:
        return

    metrics = simulation["metrics"]

    st.subheader("Simulation Results")

    c1, c2, c3 = st.columns(3)

    c1.metric("Mean", f"{metrics['mean']:,.4f}")
    c2.metric("Median", f"{metrics['median']:,.4f}")
    c3.metric("Standard Deviation", f"{metrics['std']:,.4f}")

    c4, c5, c6 = st.columns(3)

    c4.metric("5th Percentile", f"{metrics['p05']:,.4f}")
    c5.metric("95th Percentile", f"{metrics['p95']:,.4f}")
    c6.metric("Simulations", f"{metrics['simulations']:,}")

    st.subheader("Threshold Risk")

    st.metric(
        "Probability of exceeding threshold",
        f"{metrics['probability_above_threshold'] * 100:.2f}%",
    )

    st.metric(
        "Probability of falling below threshold",
        f"{metrics['probability_below_threshold'] * 100:.2f}%",
    )

    st.subheader("Prediction Distribution")

    predictions = simulation["predictions"]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        predictions,
        bins=50,
        edgecolor="black",
        alpha=0.75,
    )

    ax.axvline(
        metrics["mean"],
        linestyle="--",
        label="Mean",
    )

    ax.axvline(
        metrics["p05"],
        linestyle=":",
        label="5th percentile",
    )

    ax.axvline(
        metrics["p95"],
        linestyle=":",
        label="95th percentile",
    )

    ax.set_title("Monte Carlo Prediction Distribution")
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Frequency")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Simulation Data")

    st.dataframe(
        simulation["results"].head(100),
        use_container_width=True,
    )

    csv = simulation["results"].to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Simulation Results",
        data=csv,
        file_name="helios_monte_carlo_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.warning(
        "These results represent input uncertainty under the "
        "specified distributions. They do not include model "
        "parameter uncertainty or prove causal relationships."
    )
