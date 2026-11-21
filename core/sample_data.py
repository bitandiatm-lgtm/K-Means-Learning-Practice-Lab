"""Sample dataset generation for the customer segmentation example.

No cluster labels are included; all clustering is computed at runtime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def get_sample_customer_df(seed: int = 42) -> pd.DataFrame:
    """Return a realistic synthetic customer dataset (no cluster labels)."""
    rng = np.random.default_rng(seed)
    n = 200

    # Create several natural groups in feature space, but do not store labels.
    sizes = [50, 45, 55, 50]
    ages = np.concatenate(
        [
            rng.normal(24, 4, sizes[0]),
            rng.normal(35, 6, sizes[1]),
            rng.normal(48, 7, sizes[2]),
            rng.normal(62, 6, sizes[3]),
        ]
    )
    income = np.concatenate(
        [
            rng.normal(28, 6, sizes[0]),
            rng.normal(72, 10, sizes[1]),
            rng.normal(95, 12, sizes[2]),
            rng.normal(45, 8, sizes[3]),
        ]
    )
    spending = np.concatenate(
        [
            rng.normal(72, 10, sizes[0]),
            rng.normal(35, 8, sizes[1]),
            rng.normal(78, 9, sizes[2]),
            rng.normal(22, 7, sizes[3]),
        ]
    )

    df = pd.DataFrame(
        {
            "CustomerID": np.arange(1, n + 1),
            "Age": np.clip(ages, 18, 80).round(0).astype(int),
            "Annual Income": np.clip(income, 10, 150).round(1),
            "Spending Score": np.clip(spending, 1, 100).round(0).astype(int),
        }
    )
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def ensure_sample_csv(path: str | Path | None = None) -> Path:
    """Write the sample CSV to disk and return its path."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "sample_customers.csv"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = get_sample_customer_df()
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    p = ensure_sample_csv()
    print(f"Wrote {p}")
