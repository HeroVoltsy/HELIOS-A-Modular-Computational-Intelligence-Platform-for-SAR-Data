
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_DAYS = 1095
HORIZON_DAYS = 7

OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "helios_cpcv_dataset.csv"


def generate_dataset():

    rng = np.random.default_rng(SEED)

    dates = pd.date_range(
        start="2023-01-01",
        periods=N_DAYS,
        freq="D",
    )

    t = np.arange(N_DAYS)

    # -----------------------------------------
    # TEMPORAL FEATURES
    # -----------------------------------------

    seasonality = (
        1
        + 0.15 * np.sin(2 * np.pi * t / 365.25)
        + 0.05 * np.sin(2 * np.pi * t / 7)
    )

    trend = 1 + 0.0003 * t

    marketing_spend = (
        5000 * seasonality * trend
        + rng.normal(0, 500, N_DAYS)
    )

    operating_cost = (
        9000 * trend
        + rng.normal(0, 800, N_DAYS)
    )

    employees = (
        35
        + 5 * np.sin(2 * np.pi * t / 365.25)
        + rng.normal(0, 3, N_DAYS)
    )

    employees = np.clip(
        employees,
        10,
        None,
    )

    price = (
        100
        + 8 * np.sin(2 * np.pi * t / 180)
        + rng.normal(0, 5, N_DAYS)
    )

    units_sold = (
        450 * seasonality
        - 0.8 * price
        + 0.015 * marketing_spend
        + rng.normal(0, 25, N_DAYS)
    )

    units_sold = np.clip(
        units_sold,
        50,
        None,
    )

    # -----------------------------------------
    # DAILY REVENUE
    # -----------------------------------------

    daily_revenue = (
        units_sold * price
        + 0.8 * marketing_spend
        - 0.3 * operating_cost
        + 1500 * seasonality
        + rng.normal(0, 2500, N_DAYS)
    )

    # -----------------------------------------
    # FUTURE TARGET
    # -----------------------------------------

    # The target for day t is the total revenue
    # from day t+1 through day t+7.

    future_revenue = np.array([
        daily_revenue[
            i + 1:i + 1 + HORIZON_DAYS
        ].sum()
        if i + HORIZON_DAYS < N_DAYS
        else np.nan
        for i in range(N_DAYS)
    ])

    # Last HORIZON_DAYS observations do not have
    # a complete future target and are excluded.

    valid = np.isfinite(future_revenue)

    dates = dates[valid]

    marketing_spend = marketing_spend[valid]
    operating_cost = operating_cost[valid]
    employees = employees[valid]
    price = price[valid]
    units_sold = units_sold[valid]

    future_revenue = future_revenue[valid]

    # -----------------------------------------
    # INFORMATION INTERVALS
    # -----------------------------------------

    start_time = dates

    end_time = dates + pd.Timedelta(
        days=HORIZON_DAYS
    )

    # -----------------------------------------
    # FINAL DATASET
    # -----------------------------------------

    df = pd.DataFrame({
        "date": dates,
        "start_time": start_time,
        "end_time": end_time,
        "marketing_spend": marketing_spend,
        "operating_cost": operating_cost,
        "employees": employees,
        "price": price,
        "units_sold": units_sold,
        "future_7d_revenue": future_revenue,
    })

    # -----------------------------------------
    # VALIDATION
    # -----------------------------------------

    assert df["start_time"].notna().all()
    assert df["end_time"].notna().all()

    assert (
        df["end_time"] >= df["start_time"]
    ).all()

    assert df["start_time"].is_monotonic_increasing

    assert not df.isna().any().any()

    assert len(df) > 0

    # -----------------------------------------
    # EXPORT
    # -----------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("=" * 55)
    print("HELIOS CPCV DATASET GENERATED")
    print("=" * 55)

    print(f"Output: {OUTPUT_FILE}")
    print(f"Observations: {len(df)}")
    print(f"Features: {len(df.columns)}")
    print(f"Prediction horizon: {HORIZON_DAYS} days")

    print(
        "Interval validation:",
        "PASS"
    )

    print(
        "Chronological ordering:",
        "PASS"
    )

    print(
        "Missing values:",
        int(df.isna().sum().sum()),
    )

    print("\nDataset preview:")
    print(df.head())

    print("\nInformation interval preview:")
    print(
        df[
            ["start_time", "end_time"]
        ].head()
    )

    return df


if __name__ == "__main__":
    generate_dataset()
