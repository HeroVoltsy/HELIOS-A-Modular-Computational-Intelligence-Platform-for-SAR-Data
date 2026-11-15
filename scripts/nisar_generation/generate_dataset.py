
from pathlib import Path

import numpy as np
import pandas as pd


def generate_dataset(n_samples=1000, seed=42):

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    dates = pd.date_range(
        start="2024-01-01",
        periods=n_samples,
        freq="D",
    )

    # --------------------------------------------------------
    # PREDICTORS
    # --------------------------------------------------------

    marketing_spend = rng.normal(
        loc=5000,
        scale=1200,
        size=n_samples,
    ).clip(500, None)

    employees = rng.integers(
        5,
        80,
        size=n_samples,
    )

    operating_cost = rng.normal(
        loc=12000,
        scale=2500,
        size=n_samples,
    ).clip(3000, None)

    price = rng.normal(
        loc=100,
        scale=15,
        size=n_samples,
    ).clip(40, None)

    # Marketing and price influence sales volume.
    units_sold = (
        300
        + 0.035 * marketing_spend
        - 1.2 * price
        + 2.5 * employees
        + rng.normal(0, 35, n_samples)
    ).clip(20, None)

    # --------------------------------------------------------
    # CATEGORICAL VARIABLES
    # --------------------------------------------------------

    regions = rng.choice(
        ["North", "South", "East", "West"],
        size=n_samples,
        p=[0.25, 0.25, 0.25, 0.25],
    )

    product_categories = rng.choice(
        ["Electronics", "Food", "Clothing"],
        size=n_samples,
    )

    # --------------------------------------------------------
    # TARGET VARIABLE
    # --------------------------------------------------------

    seasonal_effect = (
        1500 * np.sin(
            2 * np.pi * np.arange(n_samples) / 365
        )
    )

    region_effect = np.select(
        [
            regions == "North",
            regions == "South",
            regions == "East",
            regions == "West",
        ],
        [1000, -500, 700, 200],
    )

    revenue = (
        12000
        + 8 * marketing_spend
        + 55 * employees
        + 3 * units_sold * price
        - 0.65 * operating_cost
        + seasonal_effect
        + region_effect
        + rng.normal(0, 5000, n_samples)
    )

    revenue = revenue.clip(0, None)

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    df = pd.DataFrame({
        "date": dates,
        "marketing_spend": marketing_spend,
        "operating_cost": operating_cost,
        "employees": employees,
        "price": price,
        "units_sold": units_sold,
        "region": regions,
        "product_category": product_categories,
        "revenue": revenue,
    })

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    missing_columns = [
        "marketing_spend",
        "operating_cost",
        "employees",
        "price",
        "region",
    ]

    for column in missing_columns:

        missing_indices = rng.choice(
            df.index,
            size=int(n_samples * 0.02),
            replace=False,
        )

        df.loc[missing_indices, column] = np.nan

    # --------------------------------------------------------
    # DUPLICATE ROWS
    # --------------------------------------------------------

    duplicate_indices = rng.choice(
        df.index,
        size=10,
        replace=False,
    )

    duplicates = df.loc[duplicate_indices].copy()

    df = pd.concat(
        [df, duplicates],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    return df


if __name__ == "__main__":

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    df = generate_dataset(
        n_samples=1000,
        seed=42,
    )

    output_path = output_dir / "helios_test_dataset.csv"

    df.to_csv(
        output_path,
        index=False,
    )

    print("HELIOS test dataset generated.")
    print(f"Location: {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing cells: {df.isna().sum().sum():,}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")
    print("\nDataset preview:")
    print(df.head())
