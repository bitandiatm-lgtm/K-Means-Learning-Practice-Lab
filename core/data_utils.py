"""Data loading, validation, and preprocessing utilities.

All functions are UI-agnostic and return structured results/errors instead of
calling Streamlit directly.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_csv_safely(source: Any) -> tuple[pd.DataFrame | None, dict[str, Any], str | None]:
    """Load CSV from a path, bytes, or Streamlit UploadedFile-like object.

    Returns (df, info, error). info includes filename/row_count/column_count when known.
    """
    info: dict[str, Any] = {"filename": None, "row_count": 0, "column_count": 0}
    try:
        if source is None:
            return None, info, "No file provided."
        if isinstance(source, (str, Path)):
            p = Path(source)
            info["filename"] = p.name
            df = pd.read_csv(p)
        else:
            # Streamlit UploadedFile or file-like
            info["filename"] = getattr(source, "name", None)
            if hasattr(source, "getvalue"):
                data = source.getvalue()
            else:
                data = source.read()
            if not data:
                return None, info, "The uploaded file is empty."
            if isinstance(data, str):
                data = data.encode("utf-8", errors="replace")
            bio = BytesIO(data)
            try:
                df = pd.read_csv(bio)
            except UnicodeDecodeError:
                bio.seek(0)
                df = pd.read_csv(bio, encoding="latin-1")
        if df is None or df.empty:
            return df, info, "The CSV contains no rows."
        info["row_count"] = int(df.shape[0])
        info["column_count"] = int(df.shape[1])
        return df, info, None
    except pd.errors.EmptyDataError:
        return None, info, "The CSV is empty or has no readable data."
    except pd.errors.ParserError as exc:
        return None, info, f"Could not parse CSV: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, info, f"Failed to read CSV: {exc}"


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return column names with numeric dtypes."""
    if df is None or df.empty:
        return []
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages per column."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["column", "missing_count", "missing_percent"])
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    out = pd.DataFrame(
        {"column": df.columns, "missing_count": missing.values, "missing_percent": pct.values}
    )
    return out


def handle_missing_values(
    df: pd.DataFrame, numeric_cols: list[str], strategy: str
) -> tuple[pd.DataFrame | None, int, str | None]:
    """Handle missing values for selected numeric columns.

    strategy: 'mean', 'median', or 'drop_rows'.
    Returns (clean_df, dropped_count, error). Works on a copy.
    """
    if df is None:
        return None, 0, "No data provided."
    if not numeric_cols:
        return None, 0, "Select at least one numeric feature."
    missing_cols = [c for c in numeric_cols if c not in df.columns]
    if missing_cols:
        return None, 0, f"Missing columns: {missing_cols}"
    work = df[numeric_cols].copy()
    # Coerce to numeric where possible (handles strings like 'N/A' in numeric columns)
    for c in numeric_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    before = len(work)
    dropped = 0
    if strategy == "drop_rows":
        work = work.dropna(axis=0)
        dropped = before - len(work)
    elif strategy in {"mean", "median"}:
        for c in numeric_cols:
            series = work[c]
            if series.isna().all():
                return None, 0, f"Column '{c}' has no valid numeric values."
            fill = series.mean() if strategy == "mean" else series.median()
            work[c] = series.fillna(fill)
    else:
        return None, 0, f"Unknown missing-value strategy: {strategy}"
    if work.empty:
        return None, dropped, "No valid records remain after handling missing values."
    # Guard against infinite values
    work = work.replace([np.inf, -np.inf], np.nan)
    if work.isna().any().any():
        if strategy == "drop_rows":
            b2 = len(work)
            work = work.dropna()
            dropped += b2 - len(work)
        else:
            for c in numeric_cols:
                if work[c].isna().all():
                    return None, dropped, f"Column '{c}' has no valid numeric values."
                work[c] = work[c].fillna(work[c].median())
    if work.empty:
        return None, dropped, "No valid records remain after cleaning."
    return work, dropped, None


def scale_features(
    X_df: pd.DataFrame, enabled: bool = True
) -> tuple[np.ndarray, StandardScaler | None]:
    """Optionally scale features with StandardScaler. Returns (array, scaler)."""
    if X_df is None or X_df.empty:
        raise ValueError("No features to scale.")
    arr = X_df.to_numpy(dtype=float)
    if not enabled:
        return arr, None
    scaler = StandardScaler()
    return scaler.fit_transform(arr), scaler


def validate_clustering_inputs(
    X: np.ndarray | pd.DataFrame,
    k: int,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int | None = 42,
) -> tuple[bool, str | None]:
    """Validate inputs for KMeans training."""
    try:
        n_samples = int(X.shape[0])
    except Exception:
        return False, "Invalid feature matrix."
    if n_samples < 2:
        return False, "At least 2 valid records are required for clustering."
    try:
        k = int(k)
        n_init = int(n_init)
        max_iter = int(max_iter)
    except Exception:
        return False, "K, n_init, and max_iter must be integers."
    if k < 2:
        return False, "K must be at least 2."
    if k > n_samples:
        return False, f"K ({k}) cannot exceed the number of valid records ({n_samples})."
    if n_init < 1:
        return False, "n_init must be at least 1."
    if max_iter < 1:
        return False, "max_iter must be at least 1."
    if random_state is not None:
        try:
            int(random_state)
        except Exception:
            return False, "random_state must be an integer or None."
    return True, None
