
from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("data/nasa_nisar")
OUTPUT_DIR = DATA_DIR / "csv"

SAMPLE_STRIDE = 50

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = list(DATA_DIR.glob("*.h5"))

if not files:
    raise FileNotFoundError("No NISAR HDF5 file found.")

H5_FILE = files[0]


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), default=str)

    if isinstance(value, np.generic):
        return value.item()

    return value


def read_value(h5, path, default=None):
    try:
        return decode(h5[path][()])
    except (KeyError, TypeError, ValueError):
        return default


def read_attributes(dataset):
    return {
        str(key): decode(value)
        for key, value in dataset.attrs.items()
    }


def quality_record(frequency, metric, value):
    return {
        "frequency": frequency,
        "metric": metric,
        "value": value,
    }


# --------------------------------------------------
# EXTRACTION
# --------------------------------------------------

def extract_frequency(h5, frequency):

    base = (
        f"/science/LSAR/GCOV/grids/{frequency}"
    )

    print(f"\nProcessing {frequency}...")

    group = h5[base]

    # Actual product datasets
    hh_ds = group["HHHH"]
    hv_ds = group["HVHV"]

    mask_ds = group["mask"]
    exception_ds = group["inputDataExceptionMask"]
    looks_ds = group["numberOfLooks"]
    rtc_ds = group["rtcGammaToSigmaFactor"]

    # Coordinate axes
    x = group["xCoordinates"][::SAMPLE_STRIDE]
    y = group["yCoordinates"][::SAMPLE_STRIDE]

    # Sample raster arrays without loading the full raster
    hh = np.asarray(
        hh_ds[::SAMPLE_STRIDE, ::SAMPLE_STRIDE],
        dtype=np.float32,
    )

    hv = np.asarray(
        hv_ds[::SAMPLE_STRIDE, ::SAMPLE_STRIDE],
        dtype=np.float32,
    )

    mask = np.asarray(
        mask_ds[::SAMPLE_STRIDE, ::SAMPLE_STRIDE]
    )

    exception = np.asarray(
        exception_ds[::SAMPLE_STRIDE, ::SAMPLE_STRIDE]
    )

    looks = np.asarray(
        looks_ds[::SAMPLE_STRIDE, ::SAMPLE_STRIDE],
        dtype=np.float32,
    )

    rtc = np.asarray(
        rtc_ds[::SAMPLE_STRIDE, ::SAMPLE_STRIDE],
        dtype=np.float32,
    )

    if not (
        hh.shape == hv.shape == mask.shape
        == exception.shape == looks.shape == rtc.shape
    ):
        raise ValueError(
            f"{frequency}: raster dimensions do not match."
        )

    if hh.shape != (len(y), len(x)):
        raise ValueError(
            f"{frequency}: coordinate dimensions do not match raster."
        )

    # Projected coordinates
    xx, yy = np.meshgrid(x, y)

    epsg = int(group["projection"][()])

    source_crs = CRS.from_epsg(epsg)

    transformer = Transformer.from_crs(
        source_crs,
        CRS.from_epsg(4326),
        always_xy=True,
    )

    longitude, latitude = transformer.transform(
        xx.ravel(),
        yy.ravel(),
    )

    # Flatten sampled arrays
    hh_flat = hh.ravel()
    hv_flat = hv.ravel()

    # Numeric validity only; preserve original masks separately.
    numeric_valid = (
        np.isfinite(hh_flat)
        & np.isfinite(hv_flat)
        & (hh_flat > 0)
        & (hv_flat > 0)
    )

    # Convert positive linear power values to dB.
    # Original measurements are retained unchanged.
    with np.errstate(divide="ignore", invalid="ignore"):
        hh_db = np.where(
            hh_flat > 0,
            10.0 * np.log10(hh_flat),
            np.nan,
        )

        hv_db = np.where(
            hv_flat > 0,
            10.0 * np.log10(hv_flat),
            np.nan,
        )

    rows, cols = np.indices(hh.shape)

    df = pd.DataFrame({
        "frequency": frequency,
        "row_index": (
            rows.ravel() * SAMPLE_STRIDE
        ),
        "column_index": (
            cols.ravel() * SAMPLE_STRIDE
        ),
        "x_projected": xx.ravel(),
        "y_projected": yy.ravel(),
        "longitude": longitude,
        "latitude": latitude,
        "HHHH": hh_flat,
        "HVHV": hv_flat,
        "HHHH_dB": hh_db,
        "HVHV_dB": hv_db,
        "number_of_looks": looks.ravel(),
        "rtc_gamma_to_sigma_factor": rtc.ravel(),
        "mask_code": mask.ravel(),
        "input_data_exception_mask": exception.ravel(),
        "numeric_measurements_valid": numeric_valid,
        "sample_stride_pixels": SAMPLE_STRIDE,
    })

    output_file = (
        OUTPUT_DIR / f"nisar_{frequency}_sample.csv"
    )

    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"EPSG: {epsg}")
    print(f"Numeric-valid samples: {numeric_valid.sum():,}")

    # --------------------------------------------------
    # QUALITY SUMMARY
    # --------------------------------------------------

    quality = []

    quality.append(
        quality_record(
            frequency,
            "sampled_pixels",
            len(df),
        )
    )

    quality.append(
        quality_record(
            frequency,
            "numeric_measurements_valid",
            int(numeric_valid.sum()),
        )
    )

    quality.append(
        quality_record(
            frequency,
            "numeric_measurements_invalid",
            int((~numeric_valid).sum()),
        )
    )

    quality.append(
        quality_record(
            frequency,
            "missing_HHHH",
            int(np.isnan(hh_flat).sum()),
        )
    )

    quality.append(
        quality_record(
            frequency,
            "missing_HVHV",
            int(np.isnan(hv_flat).sum()),
        )
    )

    for code, count in pd.Series(mask.ravel()).value_counts(
        dropna=False
    ).items():
        quality.append(
            quality_record(
                frequency,
                f"mask_code_{code}",
                int(count),
            )
        )

    for code, count in pd.Series(
        exception.ravel()
    ).value_counts(dropna=False).items():
        quality.append(
            quality_record(
                frequency,
                f"exception_mask_code_{code}",
                int(count),
            )
        )

    # --------------------------------------------------
    # METADATA
    # --------------------------------------------------

    metadata = {
        "source_file": H5_FILE.name,
        "frequency": frequency,
        "epsg": epsg,
        "sample_stride_pixels": SAMPLE_STRIDE,
        "original_raster_rows": hh_ds.shape[0],
        "original_raster_columns": hh_ds.shape[1],
        "sampled_rows": hh.shape[0],
        "sampled_columns": hh.shape[1],
        "HHHH_dataset": hh_ds.name,
        "HVHV_dataset": hv_ds.name,
        "HHHH_attributes": json.dumps(
            read_attributes(hh_ds),
            default=str,
        ),
        "HVHV_attributes": json.dumps(
            read_attributes(hv_ds),
            default=str,
        ),
        "list_of_polarizations": read_value(
            h5, base + "/listOfPolarizations"
        ),
        "list_of_covariance_terms": read_value(
            h5, base + "/listOfCovarianceTerms"
        ),
        "x_coordinate_spacing": read_value(
            h5, base + "/xCoordinateSpacing"
        ),
        "y_coordinate_spacing": read_value(
            h5, base + "/yCoordinateSpacing"
        ),
        "center_frequency_hz": read_value(
            h5,
            f"/science/LSAR/GCOV/metadata/sourceData/"
            f"swaths/{frequency}/centerFrequency",
        ),
        "product_level": read_value(
            h5, "/science/LSAR/identification/productLevel"
        ),
        "product_type": read_value(
            h5, "/science/LSAR/identification/productType"
        ),
        "product_version": read_value(
            h5, "/science/LSAR/identification/productVersion"
        ),
        "product_doi": read_value(
            h5, "/science/LSAR/identification/productDoi"
        ),
        "granule_id": read_value(
            h5, "/science/LSAR/identification/granuleId"
        ),
        "radar_band": read_value(
            h5, "/science/LSAR/identification/radarBand"
        ),
        "orbit_pass_direction": read_value(
            h5, "/science/LSAR/identification/orbitPassDirection"
        ),
        "acquisition_start": read_value(
            h5, "/science/LSAR/identification/zeroDopplerStartTime"
        ),
        "acquisition_end": read_value(
            h5, "/science/LSAR/identification/zeroDopplerEndTime"
        ),
        "processing_datetime": read_value(
            h5, "/science/LSAR/identification/processingDateTime"
        ),
        "processing_rtc_applied": read_value(
            h5,
            "/science/LSAR/GCOV/metadata/processingInformation/"
            "parameters/radiometricTerrainCorrectionApplied",
        ),
    }

    return df, quality, metadata


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    all_frames = []
    all_quality = []
    all_metadata = []

    print("=" * 65)
    print("HELIOS | NASA NISAR GCOV CSV EXTRACTION")
    print("=" * 65)
    print("Source:", H5_FILE.name)
    print("Sampling stride:", SAMPLE_STRIDE)

    with h5py.File(H5_FILE, "r") as h5:

        for frequency in ("frequencyA", "frequencyB"):

            base = (
                f"/science/LSAR/GCOV/grids/{frequency}"
            )

            if base not in h5:
                print(f"Skipping missing group: {frequency}")
                continue

            df, quality, metadata = extract_frequency(
                h5,
                frequency,
            )

            all_frames.append(df)
            all_quality.extend(quality)
            all_metadata.append(metadata)

    if not all_frames:
        raise RuntimeError("No frequency data extracted.")

    combined = pd.concat(
        all_frames,
        ignore_index=True,
    )

    combined.to_csv(
        OUTPUT_DIR / "nisar_backscatter_sample.csv",
        index=False,
    )

    pd.DataFrame(all_quality).to_csv(
        OUTPUT_DIR / "nisar_quality.csv",
        index=False,
    )

    pd.DataFrame(all_metadata).to_csv(
        OUTPUT_DIR / "nisar_metadata.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("EXTRACTION COMPLETE")
    print("=" * 65)

    for path in sorted(OUTPUT_DIR.glob("*.csv")):
        print(
            f"{path.name}: "
            f"{path.stat().st_size / 1e6:.2f} MB"
        )

    print("\nCombined observations:", f"{len(combined):,}")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

