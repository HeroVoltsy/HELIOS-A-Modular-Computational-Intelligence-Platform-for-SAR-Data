
import streamlit as st
import pandas as pd

from modules.validation.engine import run_purged_cpcv


def get_datetime_columns(df):
    """Identify columns containing valid datetime values."""

    datetime_columns = []

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(column)
            continue

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            parsed = pd.to_datetime(
                series,
                errors="coerce",
                utc=True,
            )

            if parsed.notna().all():
                datetime_columns.append(column)

    return datetime_columns


def render_validation(result, df):

    st.title("Purged Cross-Validation")

    st.caption(
        "Combinatorial validation with interval purging "
        "and a configurable embargo."
    )

    if result is None:
        st.warning("Train a regression model first.")
        return

    if df is None:
        st.warning("The original dataset is required.")
        return

    if "X_train" not in result:
        st.error(
            "Training predictors are missing from the model result. "
            "Refit the regression model."
        )
        return

    X_train = result["X_train"]

    if not X_train.index.isin(df.index).all():
        st.error(
            "Training indices do not align with the original dataset."
        )
        return

    st.subheader("Information Interval Configuration")

    datetime_columns = get_datetime_columns(df)

    if len(datetime_columns) < 2:
        st.warning(
            "CPCV requires two valid timestamp columns: "
            "an information interval start and end."
        )

        st.info(
            "This dataset currently has only one date column. "
            "A valid end-time column must be defined from the "
            "actual target information horizon. HELIOS will not "
            "automatically invent an interval."
        )

        st.caption(
            "You can still use the regression holdout metrics "
            "for this dataset."
        )

        return

    start_col = st.selectbox(
        "Information interval start column",
        datetime_columns,
        key="cpcv_start_col",
    )

    end_options = [
        column
        for column in datetime_columns
        if column != start_col
    ]

    end_col = st.selectbox(
        "Information interval end column",
        end_options,
        key="cpcv_end_col",
    )

    st.caption(
        "Intervals must describe the actual information period "
        "associated with each observation."
    )

    st.subheader("CPCV Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        n_groups = st.number_input(
            "Number of groups",
            min_value=3,
            max_value=20,
            value=5,
            step=1,
        )

    with col2:
        n_test_groups = st.number_input(
            "Validation groups",
            min_value=1,
            max_value=int(n_groups) - 1,
            value=2,
            step=1,
        )

    with col3:
        embargo = st.text_input(
            "Embargo duration",
            value="0D",
            help="Examples: 1D, 12h, 30min.",
        )

    if st.button(
        "Run Purged CPCV",
        type="primary",
        use_container_width=True,
    ):

        try:
            sample_info = df[
                [start_col, end_col]
            ].copy()

            sample_info = sample_info.rename(
                columns={
                    start_col: "start_time",
                    end_col: "end_time",
                }
            )

            sample_info["start_time"] = pd.to_datetime(
                sample_info["start_time"],
                errors="coerce",
                utc=True,
            )

            sample_info["end_time"] = pd.to_datetime(
                sample_info["end_time"],
                errors="coerce",
                utc=True,
            )

            if sample_info.isna().any().any():
                raise ValueError(
                    "Selected interval columns contain invalid timestamps."
                )

            invalid = sample_info[
                sample_info["end_time"]
                < sample_info["start_time"]
            ]

            if not invalid.empty:
                raise ValueError(
                    f"{len(invalid)} observations have end times "
                    "before their start times."
                )

            with st.spinner(
                "Running combinatorial validation..."
            ):

                validation = run_purged_cpcv(
                    result=result,
                    sample_info=sample_info,
                    n_groups=int(n_groups),
                    n_test_groups=int(n_test_groups),
                    embargo=embargo,
                )

            st.session_state[
                "helios_validation_result"
            ] = validation

        except (
            ValueError,
            KeyError,
            TypeError,
        ) as error:

            st.session_state.pop(
                "helios_validation_result",
                None,
            )

            st.error(
                f"CPCV failed: {error}"
            )

    validation = st.session_state.get(
        "helios_validation_result"
    )

    if validation is None:
        return

    st.subheader("Validation Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total splits",
        validation["n_splits"],
    )

    col2.metric(
        "Completed splits",
        validation["n_completed"],
    )

    col3.metric(
        "Skipped splits",
        validation["n_skipped"],
    )

    st.metric(
        "Embargo",
        validation["embargo"],
    )

    folds = validation["results"]

    completed = folds[
        folds["Status"] == "Completed"
    ]

    if completed.empty:
        st.warning(
            "No CPCV splits completed successfully."
        )
        st.dataframe(
            folds,
            use_container_width=True,
            hide_index=True,
        )
        return

    st.subheader("Performance Summary")

    for column, label in [
        ("R2", "R²"),
        ("RMSE", "RMSE"),
        ("MAE", "MAE"),
    ]:

        values = completed[column].dropna()

        if values.empty:
            continue

        col1, col2, col3 = st.columns(3)

        col1.metric(
            f"{label} Mean",
            f"{values.mean():.4f}",
        )

        col2.metric(
            "Standard deviation",
            f"{values.std():.4f}",
        )

        col3.metric(
            "Minimum – Maximum",
            f"{values.min():.4f} – {values.max():.4f}",
        )

    st.subheader("Fold-Level Results")

    st.dataframe(
        folds,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Validation Performance")

    chart_data = completed.set_index("Split")[
        ["R2", "RMSE", "MAE"]
    ]

    st.line_chart(
        chart_data
    )

    st.subheader("Purging and Embargo")

    st.bar_chart(
        completed.set_index("Split")[
            ["Purged", "Embargoed"]
        ]
    )

    st.download_button(
        "Download CPCV Results",
        data=folds.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="helios_purged_cpcv_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.warning(
        "CPCV requires correctly specified information intervals. "
        "These results do not replace an untouched chronological "
        "test set or reconstruct CPCV backtest paths."
    )
