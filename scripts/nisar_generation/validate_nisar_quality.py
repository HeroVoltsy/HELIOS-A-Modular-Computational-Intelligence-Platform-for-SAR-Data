
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/nasa_nisar/csv/nisar_backscatter_sample.csv")
OUTPUT_DIR = Path("data/nasa_nisar/validated")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

# Preserve original quality codes and measurements.
exception = df["input_data_exception_mask"].astype("uint32")

# NISAR quality conditions for this HH/HV product.
hh_valid_bit = (exception & 256) != 0
hv_valid_bit = (exception & 512) != 0
sample_slip = (exception & 2) != 0

quality_mask_valid = df["mask_code"] == 1
exception_mask_present = df["input_data_exception_mask"] != 65535

measurements_available = (
    np.isfinite(df["HHHH"])
    & np.isfinite(df["HVHV"])
    & (df["HHHH"] >= 0)
    & (df["HVHV"] >= 0)
)

df["quality_mask_valid"] = quality_mask_valid
df["hh_valid_bit"] = hh_valid_bit
df["hv_valid_bit"] = hv_valid_bit
df["sample_slip_flag"] = sample_slip

df["product_quality_valid"] = (
    quality_mask_valid
    & exception_mask_present
    & hh_valid_bit
    & hv_valid_bit
    & ~sample_slip
    & measurements_available
)

# dB conversion requires strictly positive values.
df["HHHH_dB_valid"] = (
    df["product_quality_valid"]
    & (df["HHHH"] > 0)
)

df["HVHV_dB_valid"] = (
    df["product_quality_valid"]
    & (df["HVHV"] > 0)
)

# Recalculate dB values using the documented product formula.
df["HHHH_dB_validated"] = np.where(
    df["HHHH_dB_valid"],
    10 * np.log10(df["HHHH"].where(df["HHHH"] > 0)),
    np.nan,
)

df["HVHV_dB_validated"] = np.where(
    df["HVHV_dB_valid"],
    10 * np.log10(df["HVHV"].where(df["HVHV"] > 0)),
    np.nan,
)

# Export full annotated sample and quality-filtered subset.
annotated_path = OUTPUT_DIR / "nisar_quality_annotated.csv"
valid_path = OUTPUT_DIR / "nisar_quality_validated.csv"
summary_path = OUTPUT_DIR / "nisar_validation_summary.csv"

df.to_csv(annotated_path, index=False)

valid = df.loc[df["product_quality_valid"]].copy()
valid.to_csv(valid_path, index=False)

# Produce frequency-specific audit statistics.
summary = []

for frequency, group in df.groupby("frequency"):
    summary.append({
        "frequency": frequency,
        "sampled_observations": len(group),
        "mask_valid": int(group["quality_mask_valid"].sum()),
        "hh_valid_bit": int(group["hh_valid_bit"].sum()),
        "hv_valid_bit": int(group["hv_valid_bit"].sum()),
        "sample_slip": int(group["sample_slip_flag"].sum()),
        "product_quality_valid": int(
            group["product_quality_valid"].sum()
        ),
        "hh_db_valid": int(group["HHHH_dB_valid"].sum()),
        "hv_db_valid": int(group["HVHV_dB_valid"].sum()),
        "quality_valid_percentage": round(
            100 * group["product_quality_valid"].mean(), 3
        ),
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(summary_path, index=False)

print("=" * 68)
print("HELIOS | NISAR QUALITY VALIDATION")
print("=" * 68)

print("\nFrequency-specific validation:")
print(summary_df.to_string(index=False))

print("\nCombined observations:", len(df))
print("Quality-valid observations:", len(valid))
print(
    "Quality-valid percentage:",
    round(100 * len(valid) / len(df), 3),
)

print("\nOutput files:")
print(annotated_path)
print(valid_path)
print(summary_path)

print("\nVALIDATION COMPLETE")
