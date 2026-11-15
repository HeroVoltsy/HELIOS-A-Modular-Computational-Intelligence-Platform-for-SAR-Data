
import numpy as np
import pandas as pd


SUPPORTED_DISTRIBUTIONS = {
    "normal",
    "uniform",
    "triangular",
}


def validate_simulation_inputs(
    result,
    distributions,
    n_simulations,
):
    if result is None:
        raise ValueError("No fitted regression model was provided.")

    if "pipeline" not in result or "features" not in result:
        raise ValueError("Invalid regression result.")

    if n_simulations < 100:
        raise ValueError("Use at least 100 simulations.")

    features = result["features"]

    missing = [
        feature
        for feature in features
        if feature not in distributions
    ]

    if missing:
        raise ValueError(
            f"Missing distributions for predictors: {missing}"
        )

    for feature in features:
        config = distributions[feature]
        distribution = config.get("distribution")

        if distribution not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"Unsupported distribution for {feature}."
            )

        if distribution == "normal":
            mean = config.get("mean")
            std = config.get("std")

            if mean is None or std is None or std <= 0:
                raise ValueError(
                    f"Invalid normal distribution for {feature}."
                )

        elif distribution == "uniform":
            low = config.get("low")
            high = config.get("high")

            if low is None or high is None or low >= high:
                raise ValueError(
                    f"Invalid uniform distribution for {feature}."
                )

        elif distribution == "triangular":
            low = config.get("low")
            mode = config.get("mode")
            high = config.get("high")

            if (
                low is None
                or mode is None
                or high is None
                or not low <= mode <= high
                or low == high
            ):
                raise ValueError(
                    f"Invalid triangular distribution for {feature}."
                )


def generate_samples(
    features,
    distributions,
    n_simulations,
    random_seed=42,
):
    rng = np.random.default_rng(random_seed)

    simulated_data = {}

    for feature in features:
        config = distributions[feature]
        distribution = config["distribution"]

        if distribution == "normal":
            values = rng.normal(
                loc=config["mean"],
                scale=config["std"],
                size=n_simulations,
            )

        elif distribution == "uniform":
            values = rng.uniform(
                low=config["low"],
                high=config["high"],
                size=n_simulations,
            )

        elif distribution == "triangular":
            values = rng.triangular(
                left=config["low"],
                mode=config["mode"],
                right=config["high"],
                size=n_simulations,
            )

        else:
            raise ValueError(
                f"Unsupported distribution: {distribution}"
            )

        simulated_data[feature] = values

    return pd.DataFrame(
        simulated_data,
        columns=features,
    )


def calculate_risk_metrics(
    predictions,
    threshold=None,
):
    predictions = np.asarray(predictions, dtype=float)

    predictions = predictions[np.isfinite(predictions)]

    if len(predictions) == 0:
        raise ValueError("No valid simulation predictions.")

    metrics = {
        "simulations": len(predictions),
        "mean": float(np.mean(predictions)),
        "median": float(np.median(predictions)),
        "std": float(np.std(predictions, ddof=1)),
        "minimum": float(np.min(predictions)),
        "maximum": float(np.max(predictions)),
        "p05": float(np.percentile(predictions, 5)),
        "p25": float(np.percentile(predictions, 25)),
        "p75": float(np.percentile(predictions, 75)),
        "p95": float(np.percentile(predictions, 95)),
    }

    if threshold is not None:
        metrics["probability_above_threshold"] = float(
            np.mean(predictions >= threshold)
        )

        metrics["probability_below_threshold"] = float(
            np.mean(predictions < threshold)
        )

    return metrics


def run_monte_carlo(
    result,
    distributions,
    n_simulations=10000,
    random_seed=42,
    threshold=None,
):
    validate_simulation_inputs(
        result=result,
        distributions=distributions,
        n_simulations=n_simulations,
    )

    features = result["features"]
    pipeline = result["pipeline"]

    simulated_inputs = generate_samples(
        features=features,
        distributions=distributions,
        n_simulations=n_simulations,
        random_seed=random_seed,
    )

    predictions = pipeline.predict(simulated_inputs)

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    valid = np.isfinite(predictions)

    if not valid.any():
        raise ValueError(
            "The model generated no valid predictions."
        )

    simulated_inputs = simulated_inputs.loc[valid].reset_index(drop=True)
    predictions = predictions[valid]

    metrics = calculate_risk_metrics(
        predictions=predictions,
        threshold=threshold,
    )

    simulation_results = simulated_inputs.copy()
    simulation_results["predicted_value"] = predictions

    return {
        "metrics": metrics,
        "predictions": predictions,
        "inputs": simulated_inputs,
        "results": simulation_results,
        "threshold": threshold,
        "random_seed": random_seed,
    }
