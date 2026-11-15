
import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy import stats

from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera


def get_transformed_training_data(result):

    pipeline = result["pipeline"]

    preprocessor = pipeline.named_steps["preprocessing"]

    X_train = result["train_data"]

    transformed = preprocessor.transform(X_train)

    return np.asarray(transformed, dtype=float)


def calculate_residual_diagnostics(result):

    residuals = np.asarray(
        result["train_residuals"],
        dtype=float
    )

    transformed_X = get_transformed_training_data(result)

    if len(residuals) < 3:
        raise ValueError("At least three observations are required.")

    # Reconstruct design matrix for diagnostic tests.
    if result["include_intercept"]:

        design = sm.add_constant(
            transformed_X,
            has_constant="add"
        )

    else:

        design = transformed_X

    dw = durbin_watson(residuals)

    jb_stat, jb_pvalue, skewness, kurtosis = jarque_bera(
        residuals
    )

    bp_results = None

    if result["include_intercept"]:

        bp_lm, bp_lm_pvalue, bp_f, bp_f_pvalue = (
            het_breuschpagan(
                residuals,
                design
            )
        )

        bp_results = {
            "lm_statistic": float(bp_lm),
            "lm_pvalue": float(bp_lm_pvalue),
            "f_statistic": float(bp_f),
            "f_pvalue": float(bp_f_pvalue),
        }

    return {
        "durbin_watson": float(dw),
        "jarque_bera_statistic": float(jb_stat),
        "jarque_bera_pvalue": float(jb_pvalue),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "breusch_pagan": bp_results,
    }


def calculate_influence_diagnostics(result):

    transformed_X = get_transformed_training_data(result)

    if result["include_intercept"]:

        design = sm.add_constant(
            transformed_X,
            has_constant="add"
        )

    else:

        design = transformed_X

    diagnostic_model = sm.OLS(
        result["y_train"],
        design
    ).fit()

    influence = diagnostic_model.get_influence()

    leverage = influence.hat_matrix_diag

    cooks_distance = influence.cooks_distance[0]

    standardized_residuals = (
        influence.resid_studentized_internal
    )

    n = len(result["y_train"])
    p = design.shape[1]

    frame = pd.DataFrame({

        "Original Index": (
            result["train_data"].index.astype(str)
        ),

        "Leverage": leverage,

        "Cook's Distance": cooks_distance,

        "Standardized Residual": standardized_residuals,

    })

    frame["High Leverage"] = (
        frame["Leverage"] > 2 * p / n
    )

    frame["Potentially Influential"] = (
        frame["Cook's Distance"] > 4 / n
    )

    return {
        "observations": frame,
        "leverage_threshold": 2 * p / n,
        "cooks_threshold": 4 / n,
    }


def calculate_qq_data(result):

    residuals = np.asarray(
        result["train_residuals"]
    )

    theoretical, ordered = stats.probplot(
        residuals,
        dist="norm"
    )[0]

    slope, intercept, correlation = stats.probplot(
        residuals,
        dist="norm"
    )[1]

    return {
        "theoretical": theoretical,
        "ordered": ordered,
        "slope": slope,
        "intercept": intercept,
        "correlation": correlation,
    }
