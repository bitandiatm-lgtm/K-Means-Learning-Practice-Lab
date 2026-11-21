"""Evaluation metrics for K-Means (inertia, silhouette, recommendations)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_inertia_range(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int | None = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> list[dict[str, Any]]:
    """Compute inertia for a range of K values."""
    n = X.shape[0]
    k_min = max(1, int(k_min))
    k_max = min(int(k_max), n)
    results: list[dict[str, Any]] = []
    for k in range(k_min, k_max + 1):
        if k < 1 or k > n:
            continue
        try:
            model = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init=int(n_init),
                max_iter=int(max_iter),
            )
            model.fit(X)
            results.append({"k": int(k), "inertia": float(model.inertia_)})
        except Exception as exc:  # noqa: BLE001
            results.append({"k": int(k), "inertia": None, "error": str(exc)})
    return results


def compute_silhouette_range(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int | None = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> list[dict[str, Any]]:
    """Compute silhouette scores for valid K values (2 <= k <= n_samples-1)."""
    n = X.shape[0]
    k_min = max(2, int(k_min))
    k_max = min(int(k_max), n - 1)
    results: list[dict[str, Any]] = []
    if n < 3:
        return results
    for k in range(k_min, k_max + 1):
        try:
            model = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init=int(n_init),
                max_iter=int(max_iter),
            )
            labels = model.fit_predict(X)
            if len(set(labels)) < 2:
                results.append({"k": int(k), "silhouette": None, "error": "Single cluster produced."})
                continue
            score = silhouette_score(X, labels)
            results.append({"k": int(k), "silhouette": float(score)})
        except Exception as exc:  # noqa: BLE001
            results.append({"k": int(k), "silhouette": None, "error": str(exc)})
    return results


def recommend_k(
    inertia_results: list[dict[str, Any]],
    silhouette_results: list[dict[str, Any]],
) -> tuple[int | None, str]:
    """Suggest a possible K using simple, explainable heuristics."""
    valid_sil = [r for r in silhouette_results if r.get("silhouette") is not None]
    if not valid_sil:
        valid_inertia = [r for r in inertia_results if r.get("inertia") is not None and r.get("k", 0) >= 2]
        if not valid_inertia:
            return None, "Not enough valid results to recommend K. Try more records or a different K range."
        # Fallback: smallest K among reasonable inertia values (conservative)
        k = int(valid_inertia[0]["k"])
        return k, (
            f"Suggested K={k} as a conservative starting point because silhouette scores were not available. "
            "Use the elbow plot to see where inertia improvements slow down."
        )

    best = max(valid_sil, key=lambda r: r["silhouette"])
    best_k = int(best["k"])
    best_score = float(best["silhouette"])
    explanation = (
        f"Suggested K={best_k} because it has the highest silhouette score ({best_score:.3f}) "
        "among the tested values. A higher silhouette score means clusters are more separated and internally cohesive. "
        "Also review the elbow plot: if inertia stops dropping sharply before or around this K, that supports the choice."
    )
    return best_k, explanation
