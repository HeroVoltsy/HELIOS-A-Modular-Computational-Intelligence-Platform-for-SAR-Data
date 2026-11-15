
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def calculate_metrics(actual, predicted, parameter_count):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    residuals = actual - predicted

    rmse = float(
        np.sqrt(mean_squared_error(actual, predicted))
    )

    mae = float(
        mean_absolute_error(actual, predicted)
    )

    r2 = float(
        r2_score(actual, predicted)
    ) if len(actual) > 1 else np.nan

    adjusted_r2 = (
        1 - (1 - r2) * (len(actual) - 1)
        / (len(actual) - parameter_count)
        if np.isfinite(r2) and len(actual) > parameter_count
        else np.nan
    )

    target_range = float(
        np.max(actual) - np.min(actual)
    )

    nrmse = (
        rmse / target_range
        if target_range > 0
        else np.nan
    )

    nonzero_mask = np.abs(actual) > 1e-8

    mape = (
        float(
            np.mean(
                np.abs(
                    residuals[nonzero_mask]
                    / actual[nonzero_mask]
                )
            ) * 100
        )
        if np.any(nonzero_mask)
        else np.nan
    )

    return {
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "rmse": rmse,
        "mae": mae,
        "nrmse": nrmse,
        "mape": mape,
    }


def build_pipeline(features, scale_features=True):

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                        if scale_features
                        else "passthrough"
                    ),
                ]),
                features,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("regression", LinearRegression()),
    ])

    return pipeline


def run_ols(
    df,
    target,
    features,
    test_fraction=0.2,
    random_seed=42,
    include_intercept=True,
    scale_features=True,
):

    if not features:
        raise ValueError("At least one predictor is required.")

    if target in features:
        raise ValueError("Target cannot also be a predictor.")

    if not 0 <= test_fraction <= 0.5:
        raise ValueError("Invalid holdout fraction.")

    selected = df[[target] + features].copy()

    for column in [target] + features:

        if not pd.api.types.is_numeric_dtype(selected[column]):
            raise ValueError(
                f"Column '{column}' must be numeric."
            )

    selected = selected.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Target cannot be imputed: remove missing targets.
    selected = selected.dropna(subset=[target])

    X = selected[features]
    y = selected[target].astype(float)

    if y.nunique() < 2:
        raise ValueError(
            "Target must contain at least two unique values."
        )

    if len(selected) < 5:
        raise ValueError("Not enough observations.")

    # Split BEFORE fitting preprocessing.
    if test_fraction > 0:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_fraction,
            random_state=random_seed,
        )
        empty_features = X_train.columns[
            X_train.isna().all()
        ].tolist()

        if empty_features:
            raise ValueError(
                f"Predictors entirely missing in training data: "
                f"{empty_features}"
            )
    else:

        X_train = X
        y_train = y
        X_test = None
        y_test = None

    parameter_count = len(features) + int(include_intercept)

    if len(X_train) <= parameter_count:
        raise ValueError(
            "Insufficient training observations."
        )

    # Build preprocessing + regression pipeline.
    pipeline = build_pipeline(
        features,
        scale_features=scale_features,
    )

    pipeline.set_params(
        regression__fit_intercept=include_intercept
    )

    # Imputation and scaling are learned only from X_train.
    pipeline.fit(X_train, y_train)

    train_predictions = pipeline.predict(X_train)

    test_predictions = (
        pipeline.predict(X_test)
        if X_test is not None
        else None
    )

    train_metrics = calculate_metrics(
        y_train,
        train_predictions,
        parameter_count,
    )

    test_metrics = (
        calculate_metrics(
            y_test,
            test_predictions,
            parameter_count,
        )
        if y_test is not None
        else None
    )

    # Recover coefficients in the original feature scale.
    preprocessor = pipeline.named_steps["preprocessing"]
    estimator = pipeline.named_steps["regression"]

    scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]

    coefficients = np.asarray(estimator.coef_, dtype=float)

    if scale_features:

        coefficients = coefficients / scaler.scale_

        intercept = (
            float(estimator.intercept_)
            - np.sum(
                coefficients * imputer.statistics_
            )
        ) if include_intercept else 0.0

    else:

        intercept = float(estimator.intercept_)

    coefficient_table = pd.DataFrame({
        "Term": ["Intercept"] + features,
        "Coefficient": [intercept] + coefficients.tolist(),
    })

    train_predictions_df = pd.DataFrame({
        "Original Index": X_train.index.astype(str),
        "Observed": y_train.to_numpy(),
        "Predicted": train_predictions,
        "Residual": y_train.to_numpy() - train_predictions,
        "Split": "Train",
    })

    test_predictions_df = None

    if X_test is not None:

        test_predictions_df = pd.DataFrame({
            "Original Index": X_test.index.astype(str),
            "Observed": y_test.to_numpy(),
            "Predicted": test_predictions,
            "Residual": y_test.to_numpy() - test_predictions,
            "Split": "Test",
        })

    return {
        "pipeline": pipeline,
        "model": pipeline,
        "target": target,
        "features": features,
        "include_intercept": include_intercept,
        "scale_features": scale_features,
        "n_total": len(selected),
        "n_train": len(X_train),
        "n_test": len(X_test) if X_test is not None else 0,
        "n_excluded": len(df) - len(selected),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "coefficients": coefficient_table,
        "train_data": X_train,
        "X_train": X_train.copy(),
        "test_data": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
        "train_residuals": y_train.to_numpy() - train_predictions,
        "train_predictions_df": train_predictions_df,
        "test_predictions_df": test_predictions_df,
        "parameter_count": parameter_count,
    }


def calculate_sensitivity(
    result,
    perturbation=0.10,
):

    if perturbation <= 0:
        raise ValueError(
            "Perturbation must be greater than zero."
        )

    pipeline = result["pipeline"]
    X_reference = result["train_data"].copy()

    baseline = pipeline.predict(X_reference)

    rows = []

    for feature in result["features"]:

        perturbed = X_reference.copy()

        scale = float(
            X_reference[feature].std()
        )

        if not np.isfinite(scale) or scale == 0:
            rows.append({
                "Feature": feature,
                "Mean Absolute Prediction Change": 0.0,
                "Mean Prediction Change": 0.0,
                "Relative Prediction Change (%)": 0.0,
            })
            continue

        # Shift one feature at a time.
        perturbed[feature] = (
            perturbed[feature]
            + perturbation * scale
        )

        changed_predictions = pipeline.predict(
            perturbed
        )

        prediction_change = (
            changed_predictions - baseline
        )

        mean_absolute_change = float(
            np.mean(np.abs(prediction_change))
        )

        mean_change = float(
            np.mean(prediction_change)
        )

        baseline_scale = max(
            float(np.mean(np.abs(baseline))),
            1e-8
        )

        rows.append({
            "Feature": feature,
            "Mean Absolute Prediction Change": mean_absolute_change,
            "Mean Prediction Change": mean_change,
            "Relative Prediction Change (%)": (
                mean_absolute_change / baseline_scale * 100
            ),
        })

    return pd.DataFrame(rows).sort_values(
        "Mean Absolute Prediction Change",
        ascending=False,
    )
