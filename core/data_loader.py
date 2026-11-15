
import pandas as pd


SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")


def load_dataset(uploaded_file):

    if uploaded_file is None:
        raise ValueError("No file was provided.")

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):

        df = pd.read_csv(uploaded_file)

    elif filename.endswith((".xlsx", ".xls")):

        df = pd.read_excel(uploaded_file)

    else:

        raise ValueError(
            f"Unsupported file format: {filename}"
        )

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    return df


def get_numeric_columns(df):

    return df.select_dtypes(
        include="number"
    ).columns.tolist()


def get_dataset_metadata(df):

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": len(get_numeric_columns(df)),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
