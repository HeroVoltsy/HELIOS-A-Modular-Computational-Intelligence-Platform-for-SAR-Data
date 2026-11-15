
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data/nasa_nisar/validated/nisar_quality_validated.csv"
)

OUTPUT_DIR = BASE_DIR / "data/nasa_nisar/helios"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# NISAR PRODUCT METADATA
# ============================================================

PRODUCT_METADATA = {
    "acquisition_date": "2026-10-21",
    "acquisition_start_utc": "2026-10-21T21:28:20Z",
    "acquisition_end_utc": "2026-10-21T21:28:52.999342105Z",

    "acquisition_start_pht": "2026-10-22T05:28:20+08:00",
    "acquisition_end_pht": "2026-10-22T05:28:52.999342105+08:00",

    "processing_datetime_utc": "2026-11-07T15:38:24Z",

    "orbit_pass_direction": "Ascending",
    "radar_band": "L",
    "product_type": "GCOV",
    "product_level": "L2",
    "product_version": "1.0.20",

    "product_doi": "10.5067/NIL2GCOV-V1",
    "projection_epsg": "32651",
    "rtc_applied": True,
}


# ============================================================
# HELIOS DATASET SCHEMA
# ============================================================

BASE_COLUMNS = [
    "frequency",

    "row_index",
    "column_index",

    "x_projected",
    "y_projected",

    "longitude",
    "latitude",

    "HHHH",
    "HVHV",

    "HHHH_dB_validated",
    "HVHV_dB_validated",

    "number_of_looks",
    "rtc_gamma_to_sigma_factor",

    "mask_code",
    "input_data_exception_mask",

    "sample_stride_pixels",
]

METADATA_COLUMNS = list(PRODUCT_METADATA.keys())

OUTPUT_COLUMNS = BASE_COLUMNS + METADATA_COLUMNS


# ============================================================
# LOAD VALIDATED OBSERVATIONS
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    print("Loading validated NISAR observations...")

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    print(f"Loaded observations: {len(df):,}")

    required_columns = (
        BASE_COLUMNS
        + ["product_quality_valid"]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# ============================================================
# PREPARE FREQUENCY DATASET
# ============================================================

def prepare_frequency(df, frequency):

    print(f"\n{'=' * 60}")
    print(f"Preparing {frequency}")
    print(f"{'=' * 60}")

    subset = df[
        (df["frequency"] == frequency)
        & (df["product_quality_valid"] == True)
    ].copy()

    if subset.empty:
        print(f"No valid observations found for {frequency}")
        return

    # Attach acquisition and product provenance.
    for column, value in PRODUCT_METADATA.items():
        subset[column] = value

    # Preserve only the HELIOS analytical schema.
    subset = subset[OUTPUT_COLUMNS].copy()

    # Ensure numeric measurement fields are numeric.
    numeric_columns = [
        "HHHH",
        "HVHV",
        "HHHH_dB_validated",
        "HVHV_dB_validated",
        "number_of_looks",
        "rtc_gamma_to_sigma_factor",
        "longitude",
        "latitude",
        "x_projected",
        "y_projected",
    ]

    for column in numeric_columns:
        subset[column] = pd.to_numeric(
            subset[column],
            errors="coerce"
        )

    # Remove rows with invalid backscatter measurements.
    subset = subset.dropna(
        subset=[
            "HHHH",
            "HVHV",
            "HHHH_dB_validated",
            "HVHV_dB_validated",
        ]
    )

    output_file = (
        OUTPUT_DIR
        / f"{frequency.lower()}_helios.csv"
    )

    subset.to_csv(
        output_file,
        index=False
    )

    print(f"Observations: {len(subset):,}")
    print(f"Output: {output_file}")

    print("\nAcquisition metadata:")
    print(f"Date: {PRODUCT_METADATA['acquisition_date']}")
    print(
        "UTC: "
        f"{PRODUCT_METADATA['acquisition_start_utc']} "
        "to "
        f"{PRODUCT_METADATA['acquisition_end_utc']}"
    )
    print(
        "PHT: "
        f"{PRODUCT_METADATA['acquisition_start_pht']} "
        "to "
        f"{PRODUCT_METADATA['acquisition_end_pht']}"
    )

    print("\nBackscatter statistics:")

    print(
        subset[
            [
                "HHHH",
                "HVHV",
                "HHHH_dB_validated",
                "HVHV_dB_validated",
            ]
        ].describe().round(4)
    )

    return subset


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    results = {}

    for frequency in ["frequencyA", "frequencyB"]:

        result = prepare_frequency(
            df,
            frequency
        )

        if result is not None:
            results[frequency] = result

    print("\n" + "=" * 60)
    print("HELIOS DATA PREPARATION COMPLETE")
    print("=" * 60)

    for frequency, result in results.items():
        print(
            f"{frequency}: "
            f"{len(result):,} observations"
        )

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
