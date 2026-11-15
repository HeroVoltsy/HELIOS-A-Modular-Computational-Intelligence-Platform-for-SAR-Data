
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def validate_inputs(
    result,
    sample_info,
    n_groups,
    n_test_groups,
    embargo,
    max_splits,
):
    if result is None:
        raise ValueError("No fitted regression result was provided.")

    required = [
        "pipeline",
        "X_train",
        "y_train",
        "features",
    ]

    missing = [
        key for key in required
        if key not in result
    ]

    if missing:
        raise ValueError(
            f"Missing model fields: {missing}"
        )

    if not isinstance(sample_info, pd.DataFrame):
        raise TypeError(
            "sample_info must be a pandas DataFrame."
        )

    required_columns = {
        "start_time",
        "end_time",
    }

    if not required_columns.issubset(sample_info.columns):
        raise ValueError(
            "sample_info requires start_time and end_time columns."
        )

    X = result["X_train"][result["features"]]
    y = result["y_train"]

    if len(X) != len(y):
        raise ValueError(
            "X_train and y_train lengths differ."
        )

    if not X.index.is_unique or not y.index.is_unique:
        raise ValueError(
            "Training indices must be unique."
        )

    if not X.index.equals(y.index):
        raise ValueError(
            "X_train and y_train indices are not aligned."
        )

    if not sample_info.index.is_unique:
        raise ValueError(
            "Sample information indices must be unique."
        )

    if not X.index.isin(sample_info.index).all():
        raise ValueError(
            "sample_info is missing training observation indices."
        )

    if n_groups < 2:
        raise ValueError(
            "n_groups must be at least 2."
        )

    if not 1 <= n_test_groups < n_groups:
        raise ValueError(
            "n_test_groups must be between 1 and n_groups - 1."
        )

    if len(X) < n_groups * 2:
        raise ValueError(
            "Not enough observations for the requested groups."
        )

    total_splits = comb(
        n_groups,
        n_test_groups,
    )

    if total_splits > max_splits:
        raise ValueError(
            f"Requested {total_splits} combinations. "
            f"Maximum allowed is {max_splits}."
        )

    try:
        embargo = pd.Timedelta(embargo)
    except (TypeError, ValueError):
        raise ValueError(
            "Embargo must be a valid duration, such as '2D'."
        )

    if embargo < pd.Timedelta(0):
        raise ValueError(
            "Embargo cannot be negative."
        )

    return X, y, embargo


def prepare_sample_info(X, sample_info):

    info = sample_info.loc[
        X.index,
        ["start_time", "end_time"],
    ].copy()

    info["start_time"] = pd.to_datetime(
        info["start_time"],
        errors="coerce",
        utc=True,
    )

    info["end_time"] = pd.to_datetime(
        info["end_time"],
        errors="coerce",
        utc=True,
    )

    if info.isna().any().any():
        invalid = info[
            info.isna().any(axis=1)
        ]

        raise ValueError(
            "Sample intervals contain missing or invalid timestamps.\n"
            f"Examples:\n{invalid.head(10).to_string()}"
        )

    invalid = info[
        info["end_time"] < info["start_time"]
    ]

    if not invalid.empty:
        raise ValueError(
            "Invalid CPCV intervals detected.\n"
            f"Number of invalid rows: {len(invalid)}\n"
            f"Examples:\n{invalid.head(10).to_string()}"
        )

    return info.sort_values(
        "start_time",
        kind="stable",
    )


def generate_combinatorial_splits(
    n_samples,
    n_groups,
    n_test_groups,
):

    groups = np.array_split(
        np.arange(n_samples),
        n_groups,
    )

    for test_group_ids in combinations(
        range(n_groups),
        n_test_groups,
    ):

        test_indices = np.concatenate([
            groups[group_id]
            for group_id in test_group_ids
        ])

        train_indices = np.setdiff1d(
            np.arange(n_samples),
            test_indices,
        )

        yield (
            test_group_ids,
            train_indices,
            test_indices,
        )


def purge_and_embargo(
    train_indices,
    test_indices,
    sample_info,
    embargo,
):

    train_info = sample_info.iloc[
        train_indices
    ]

    test_info = sample_info.iloc[
        test_indices
    ]

    train_starts = train_info["start_time"]
    train_ends = train_info["end_time"]

    purged = np.zeros(
        len(train_indices),
        dtype=bool,
    )

    embargoed = np.zeros(
        len(train_indices),
        dtype=bool,
    )

    for test_start, test_end in zip(
        test_info["start_time"],
        test_info["end_time"],
    ):

        overlaps = (
            (train_starts <= test_end)
            & (train_ends >= test_start)
        )

        purged |= overlaps.to_numpy()

        embargo_mask = (
            (train_starts > test_end)
            & (train_starts <= test_end + embargo)
        )

        embargoed |= embargo_mask.to_numpy()

    embargoed_only = embargoed & ~purged
    removed = purged | embargoed

    retained_indices = train_indices[
        ~removed
    ]

    return {
        "train_indices": retained_indices,
        "removed_count": int(removed.sum()),
        "purged_count": int(purged.sum()),
        "embargoed_count": int(embargoed_only.sum()),
    }


def run_purged_cpcv(
    result,
    sample_info,
    n_groups=5,
    n_test_groups=2,
    embargo="0D",
    max_splits=100,
):

    X, y, embargo = validate_inputs(
        result,
        sample_info,
        n_groups,
        n_test_groups,
        embargo,
        max_splits,
    )

    info = prepare_sample_info(
        X,
        sample_info,
    )

    # Align observations to chronological interval order.
    X = X.loc[info.index]
    y = y.loc[info.index]

    rows = []

    splits = generate_combinatorial_splits(
        len(X),
        n_groups,
        n_test_groups,
    )

    for split_number, (
        test_group_ids,
        train_indices,
        test_indices,
    ) in enumerate(splits, start=1):

        filtered = purge_and_embargo(
            train_indices,
            test_indices,
            info,
            embargo,
        )

        retained_train = filtered["train_indices"]

        if len(retained_train) < 2:
            rows.append({
                "Split": split_number,
                "Test Groups": str(test_group_ids),
                "Status": "Skipped",
                "Reason": "Insufficient training observations after purging.",
                "Train Samples": len(retained_train),
                "Test Samples": len(test_indices),
                "Purged": filtered["purged_count"],
                "Embargoed": filtered["embargoed_count"],
                "R2": np.nan,
                "RMSE": np.nan,
                "MAE": np.nan,
            })
            continue

        if len(test_indices) == 0:
            continue

        X_train = X.iloc[retained_train]
        y_train = y.iloc[retained_train]

        X_test = X.iloc[test_indices]
        y_test = y.iloc[test_indices]

        pipeline = clone(
            result["pipeline"]
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        predictions = pipeline.predict(
            X_test
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions,
                )
            )
        )

        mae = float(
            mean_absolute_error(
                y_test,
                predictions,
            )
        )

        r2 = (
            float(
                r2_score(
                    y_test,
                    predictions,
                )
            )
            if len(y_test) >= 2
            else np.nan
        )

        rows.append({
            "Split": split_number,
            "Test Groups": str(test_group_ids),
            "Status": "Completed",
            "Reason": "",
            "Train Samples": len(retained_train),
            "Test Samples": len(test_indices),
            "Purged": filtered["purged_count"],
            "Embargoed": filtered["embargoed_count"],
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
        })

    results = pd.DataFrame(rows)

    if results.empty:
        raise ValueError(
            "CPCV did not produce any results."
        )

    return {
        "results": results,
        "n_splits": len(results),
        "n_completed": int(
            (results["Status"] == "Completed").sum()
        ),
        "n_skipped": int(
            (results["Status"] == "Skipped").sum()
        ),
        "mean_r2": results["R2"].mean(),
        "mean_rmse": results["RMSE"].mean(),
        "mean_mae": results["MAE"].mean(),
        "embargo": str(embargo),
        "n_groups": n_groups,
        "n_test_groups": n_test_groups,
    }
