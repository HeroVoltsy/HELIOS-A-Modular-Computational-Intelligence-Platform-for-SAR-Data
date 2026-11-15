
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path("data/nasa_nisar")
CSV_DIR = ROOT / "csv"

H5_FILE = next(ROOT.glob("*.h5"))

print("=" * 72)
print("HELIOS | NISAR DATA VALIDATION")
print("=" * 72)

# 1. Inspect extracted CSV files
for filename in [
    "nisar_metadata.csv",
    "nisar_quality.csv",
    "nisar_backscatter_sample.csv",
]:
    path = CSV_DIR / filename

    print(f"\n{'=' * 72}\n{filename}\n{'=' * 72}")

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(10).to_string(index=False))

    if filename == "nisar_backscatter_sample.csv":
        print("\nNumeric summary:")
        print(df.describe().T.to_string())

        print("\nMissing values:")
        print(df.isna().sum().to_string())

# 2. Inspect original HDF5 metadata and mask definitions
print(f"\n{'=' * 72}")
print("ORIGINAL HDF5 PRODUCT METADATA")
print("=" * 72)

with h5py.File(H5_FILE, "r") as h5:

    paths = [
        "/science/LSAR/GCOV/metadata/ceosAnalysisReadyData/"
        "outputBackscatterDecibelConversionFormula",

        "/science/LSAR/identification/granuleId",
        "/science/LSAR/identification/productLevel",
        "/science/LSAR/identification/productType",
        "/science/LSAR/identification/radarBand",
        "/science/LSAR/identification/orbitPassDirection",
        "/science/LSAR/identification/zeroDopplerStartTime",
        "/science/LSAR/identification/zeroDopplerEndTime",
    ]

    for path in paths:
        print(f"\n{path}")

        if path in h5:
            obj = h5[path]

            if isinstance(obj, h5py.Dataset):
                value = obj[()]
                print("Value:", value)
                print("Attributes:")

                for key, val in obj.attrs.items():
                    print(f"  {key}: {val}")
            else:
                print("Group")

        else:
            print("NOT FOUND")

    # 3. Inspect frequency-specific metadata and quality masks
    for frequency in ["frequencyA", "frequencyB"]:

        base = f"/science/LSAR/GCOV/grids/{frequency}"

        print(f"\n{'=' * 72}")
        print(f"{frequency.upper()} QUALITY AND RADIOMETRIC INSPECTION")
        print("=" * 72)

        for name in [
            "HHHH",
            "HVHV",
            "mask",
            "inputDataExceptionMask",
            "numberOfLooks",
            "rtcGammaToSigmaFactor",
            "projection",
            "xCoordinateSpacing",
            "yCoordinateSpacing",
        ]:
            path = f"{base}/{name}"

            if path not in h5:
                print(f"\n{name}: NOT FOUND")
                continue

            ds = h5[path]

            print(f"\n{name}")
            print("Shape:", ds.shape)
            print("Dtype:", ds.dtype)

            for key, value in ds.attrs.items():
                print(f"Attribute {key}: {value}")

            if name in ["mask", "inputDataExceptionMask"]:
                sample = ds[::50, ::50]
                values, counts = np.unique(
                    sample, return_counts=True
                )

                print("Sampled value counts:")
                for value, count in zip(values, counts):
                    print(f"  {value}: {count}")

            elif name == "projection":
                print("Projection value:", ds[()])

            elif name in ["xCoordinateSpacing", "yCoordinateSpacing"]:
                print("Coordinate spacing:", ds[()])

        # 4. Inspect polarization and covariance labels
        for name in [
            "listOfPolarizations",
            "listOfCovarianceTerms",
        ]:
            path = f"{base}/{name}"

            if path in h5:
                print(f"\n{name}:", h5[path][()])

print("\nVALIDATION INSPECTION COMPLETE")
