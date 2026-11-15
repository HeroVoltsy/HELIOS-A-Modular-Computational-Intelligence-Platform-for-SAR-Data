
import numpy as np
import pandas as pd


def _validate_model_result(result):
    if result is None:
        raise ValueError("No fitted regression model was provided.")

    if "pipeline" not in result or "features" not in result:
        raise ValueError("Invalid regression result.")

    if not result["features"]:
        raise ValueError("The model has no predictors.")

    if "X_train" not in result:
        raise ValueError(
            "Training predictors are missing from the model result."
        )


def get_baseline_values(result):
    """
    Use training-set medians as default baseline values.
    """
    _validate_model_result(result)

    X_train = result["X_train"]
    features = result["features"]

    baseline = X_train[features].median()

    if baseline.isna().any():
        invalid = baseline[baseline.isna()].index.tolist()
        raise ValueError(
            f"Cannot calculate baseline values for: {invalid}"
        )

    return baseline.to_dict()


def _build_input_frame(values, features):
    missing = [
        feature for feature in features
        if feature not in values
    ]

    if missing:
        raise ValueError(
            f"Missing predictor values: {missing}"
        )

    row = {}

    for feature in features:
        try:
            value = float(values[feature])
        except (TypeError, ValueError):
            raise ValueError(
                f"{feature} must have a numeric value."
            )

        if not np.isfinite(value):
            raise ValueError(
                f"{feature} must have a finite value."
            )

        row[feature] = value

    return pd.DataFrame([row], columns=features)


def predict_scenario(result, values):
    """
    Predict one scenario using the existing fitted Pipeline.
    """
    _validate_model_result(result)

    features = result["features"]

    input_frame = _build_input_frame(
        values=values,
        features=features,
    )

    prediction = result["pipeline"].predict(input_frame)

    prediction = float(prediction[0])

    if not np.isfinite(prediction):
        raise ValueError("The model returned an invalid prediction.")

    return prediction


def compare_scenarios(
    result,
    baseline_values,
    scenario_values,
):
    """
    Compare baseline and alternative scenario predictions.
    """
    baseline_prediction = predict_scenario(
        result,
        baseline_values,
    )

    scenario_prediction = predict_scenario(
        result,
        scenario_values,
    )

    difference = scenario_prediction - baseline_prediction

    if abs(baseline_prediction) > 1e-12:
        percentage_change = (
            difference / abs(baseline_prediction)
        ) * 100
    else:
        percentage_change = None

    feature_comparison = []

    for feature in result["features"]:
        baseline_value = float(baseline_values[feature])
        scenario_value = float(scenario_values[feature])

        feature_comparison.append({
            "feature": feature,
            "baseline": baseline_value,
            "scenario": scenario_value,
            "difference": scenario_value - baseline_value,
        })

    return {
        "baseline_prediction": baseline_prediction,
        "scenario_prediction": scenario_prediction,
        "absolute_difference": difference,
        "percentage_change": percentage_change,
        "feature_comparison": pd.DataFrame(
            feature_comparison
        ),
    }


def compare_multiple_scenarios(
    result,
    baseline_values,
    scenarios,
):
    """
    Compare multiple named scenarios against one baseline.

    scenarios format:
    {
        "Optimistic": {"feature_a": 10, "feature_b": 20},
        "Pessimistic": {"feature_a": 5, "feature_b": 30}
    }
    """
    baseline_prediction = predict_scenario(
        result,
        baseline_values,
    )

    rows = []

    for scenario_name, scenario_values in scenarios.items():
        prediction = predict_scenario(
            result,
            scenario_values,
        )

        difference = prediction - baseline_prediction

        percentage_change = (
            difference / abs(baseline_prediction) * 100
            if abs(baseline_prediction) > 1e-12
            else np.nan
        )

        rows.append({
            "scenario": scenario_name,
            "predicted_value": prediction,
            "absolute_difference": difference,
            "percentage_change": percentage_change,
        })

    return pd.DataFrame(rows)
