"""Data cleaning utilities: missing value handling and outlier capping (IQR)."""
import pandas as pd
import numpy as np


def report_missing(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum()[df.isnull().sum() > 0]


def fill_numeric_median(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    return df


def cap_outliers_iqr(df: pd.DataFrame, cols, factor: float = 1.5) -> pd.DataFrame:
    """Cap outliers using the IQR method (Section 18-19 of the spec)."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        df[c] = df[c].clip(lower=lower, upper=upper)
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()
