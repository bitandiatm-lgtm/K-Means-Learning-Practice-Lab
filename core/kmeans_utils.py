"""K-Means training and cluster reporting utilities (UI-agnostic)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def fit_kmeans(
    X: np.ndarray,
    k: int,
    random_state: int | None = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> dict[str, Any]:
    """Fit sklearn KMeans and return a structured result."""
    model = KMeans(
        n_clusters=int(k),
        random_state=random_state,
        n_init=int(n_init),
        max_iter=int(max_iter),
    )
    labels = model.fit_predict(X)
    return {
        "model": model,
        "labels": labels,
        "centroids_scaled": model.cluster_centers_,
        "inertia": float(model.inertia_),
        "n_iter": int(model.n_iter_),
        "k": int(k),
    }


def centroids_original_scale(
    centroids_scaled: np.ndarray,
    scaler: StandardScaler | None,
    feature_names: list[str],
    scaling_enabled: bool,
) -> pd.DataFrame:
    """Convert centroids back to original feature units when scaling was used."""
    if scaling_enabled and scaler is not None:
        centroids = scaler.inverse_transform(centroids_scaled)
    else:
        centroids = np.asarray(centroids_scaled, dtype=float)
    df = pd.DataFrame(centroids, columns=feature_names)
    df.insert(0, "Cluster", np.arange(df.shape[0]))
    return df


def add_labels_to_copy(
    df: pd.DataFrame, labels: np.ndarray, col_name: str = "Cluster"
) -> pd.DataFrame:
    """Return a copy of df with cluster labels added (original df unchanged)."""
    out = df.copy()
    out[col_name] = np.asarray(labels)
    return out


def cluster_counts(labels: np.ndarray) -> pd.DataFrame:
    """Return cluster record counts and percentages."""
    s = pd.Series(np.asarray(labels), name="Cluster")
    counts = s.value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(2)
    return pd.DataFrame(
        {"Cluster": counts.index.astype(int), "Count": counts.values, "Percent": pct.values}
    )


def cluster_feature_stats(
    df_labeled: pd.DataFrame, feature_cols: list[str], label_col: str = "Cluster"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute mean and median per cluster for selected features."""
    valid = [c for c in feature_cols if c in df_labeled.columns]
    if not valid:
        return pd.DataFrame(), pd.DataFrame()
    grouped = df_labeled.groupby(label_col)[valid]
    means = grouped.mean(numeric_only=True).reset_index()
    medians = grouped.median(numeric_only=True).reset_index()
    return means, medians


def describe_clusters(
    centroids_original: pd.DataFrame, feature_cols: list[str]
) -> list[str]:
    """Generate simple, data-driven explanations from centroid statistics."""
    explanations: list[str] = []
    if centroids_original.empty or not feature_cols:
        return explanations
    overall = centroids_original[feature_cols].mean(numeric_only=True)
    for _, row in centroids_original.iterrows():
        cluster = int(row["Cluster"])
        parts = []
        for feat in feature_cols:
            val = float(row[feat])
            avg = float(overall[feat])
            if avg == 0:
                relation = "around the overall average"
            elif val >= avg * 1.15:
                relation = "higher than average"
            elif val <= avg * 0.85:
                relation = "lower than average"
            else:
                relation = "near the overall average"
            parts.append(f"{feat} is {relation} ({val:.2f})")
        explanations.append(f"Cluster {cluster}: " + "; ".join(parts) + ".")
    return explanations
