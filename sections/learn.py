"""Learn K-Means section: beginner-friendly concepts and interactive demo."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
]
CENTROID_COLORS = [
    "#1d4ed8",
    "#b91c1c",
    "#15803d",
    "#7e22ce",
]


def _make_demo_data(seed: int = 7, n_per: int = 5) -> np.ndarray:
    """Generate a small, reproducible 2D dataset with visible structure."""
    rng = np.random.default_rng(seed)
    centers = np.array([[2.0, 2.2], [6.2, 2.0], [4.2, 6.0]])
    points = []
    for c in centers:
        points.append(c + rng.normal(0, 0.55, size=(n_per, 2)))
    # A couple of points to make assignment/boundary interesting.
    points.append(np.array([[3.6, 3.8], [5.0, 4.2], [4.8, 3.2]]))
    data = np.vstack(points)
    return data


def _init_centroids(data: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data), size=k, replace=False)
    return data[idx].copy()


def _assign_clusters(data: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # distances shape: (n_points, k)
    diff = data[:, None, :] - centroids[None, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    labels = np.argmin(distances, axis=1)
    return labels, distances


def _recompute_centroids(data: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    centroids = np.zeros((k, data.shape[1]))
    for j in range(k):
        mask = labels == j
        if np.any(mask):
            centroids[j] = data[mask].mean(axis=0)
        else:
            # Empty cluster: keep a safe fallback (rare in this tiny demo).
            centroids[j] = data[np.random.randint(0, len(data))]
    return centroids


def _run_kmeans_steps(
    data: np.ndarray, k: int, init_seed: int, max_iters: int = 10, tol: float = 1e-6
) -> list[dict[str, Any]]:
    """Compute step states for an interactive walkthrough.

    State layout:
    - iteration 0: initial centroids, no assignments yet.
    - iteration t>=1: assignments using previous centroids, then recalculated centroids.
    """
    centroids = _init_centroids(data, k, init_seed)
    states: list[dict[str, Any]] = [
        {
            "iteration": 0,
            "phase": "Initial centroids selected",
            "centroids": centroids.copy(),
            "labels": None,
            "distances": None,
            "converged": False,
            "inertia": None,
        }
    ]

    prev = centroids.copy()
    for it in range(1, max_iters + 1):
        labels, distances = _assign_clusters(data, prev)
        new_centroids = _recompute_centroids(data, labels, k)
        shift = float(np.max(np.sqrt(np.sum((new_centroids - prev) ** 2, axis=1))))
        inertia = float(np.sum((data - new_centroids[labels]) ** 2))
        converged = shift <= tol
        states.append(
            {
                "iteration": it,
                "phase": "Assigned points, recalculated centroids"
                + (" (converged)" if converged else ""),
                "centroids": new_centroids.copy(),
                "labels": labels.copy(),
                "distances": distances.copy(),
                "converged": converged,
                "inertia": inertia,
                "max_centroid_shift": shift,
            }
        )
        prev = new_centroids
        if converged:
            break
    return states


def _plot_state(data: np.ndarray, state: dict[str, Any], k: int) -> go.Figure:
    labels = state["labels"]
    centroids = state["centroids"]
    fig = go.Figure()

    if labels is None:
        fig.add_trace(
            go.Scatter(
                x=data[:, 0],
                y=data[:, 1],
                mode="markers",
                name="Data points",
                marker=dict(size=10, color="#64748b", line=dict(width=1, color="white")),
                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra>point</extra>",
            )
        )
    else:
        for j in range(k):
            mask = labels == j
            pts = data[mask]
            fig.add_trace(
                go.Scatter(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    mode="markers",
                    name=f"Cluster {j + 1}",
                    marker=dict(
                        size=10,
                        color=COLORS[j % len(COLORS)],
                        line=dict(width=1, color="white"),
                    ),
                    hovertemplate=(
                        f"Cluster {j + 1}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<extra></extra>"
                    ),
                )
            )

    fig.add_trace(
        go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode="markers+text",
            name="Centroids",
            text=[f"C{j + 1}" for j in range(k)],
            textposition="top center",
            marker=dict(
                size=16,
                symbol="x",
                color=[CENTROID_COLORS[j % len(CENTROID_COLORS)] for j in range(k)],
                line=dict(width=2),
            ),
            hovertemplate="Centroid %{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Iteration {state['iteration']}: {state['phase']}",
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        template="plotly_white",
        height=430,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=70, b=20),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _render_concepts() -> None:
    st.markdown(
        """
This page explains **K-Means clustering** in plain language and lets you watch the
algorithm work step by step on a tiny 2D dataset.
"""
    )

    with st.expander("Machine Learning fundamentals", expanded=True):
        st.markdown(
            """
**Machine Learning (ML)** is a way to make computers learn patterns from data instead of
following only hand-written rules.

- **Input data**: examples (rows) described by features (columns).
- **Learning**: the algorithm finds patterns, relationships, or structure.
- **Output**: predictions, groups, or insights you can use.

ML is useful when rules are hard to write explicitly, but examples are available.
"""
        )

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.subheader("Supervised learning")
            st.markdown(
                """
- Uses **labeled** data (each example has a known answer/target).
- Goal: learn a mapping from inputs to outputs.
- Typical tasks: **classification** (spam/not spam) and **regression** (price prediction).
- Evaluation compares predictions to known labels.
"""
            )
    with c2:
        with st.container(border=True):
            st.subheader("Unsupervised learning")
            st.markdown(
                """
- Uses **unlabeled** data (no target column to predict).
- Goal: discover structure hidden in the data.
- Typical tasks: **clustering**, dimensionality reduction, anomaly detection.
- Evaluation often uses internal metrics (like inertia/silhouette), not ground-truth labels.
"""
            )

    with st.expander("Clustering and K-Means", expanded=True):
        st.markdown(
            """
**Clustering** groups similar data points together so that:
- points in the same cluster are **more similar** to each other, and
- points in different clusters are **less similar**.

**K-Means** is a popular clustering algorithm. You choose **K**, the number of clusters,
and K-Means tries to find K groups by iterating these steps:

1. **Initialize** K centroids (starting center points).
2. **Assign** each point to the nearest centroid (using distance).
3. **Recalculate** each centroid as the mean of points assigned to it.
4. **Repeat** assignment and recalculation until centroids stop moving much (**convergence**).

K-Means works well when clusters are roughly compact and separated, but it requires choosing K
and can be sensitive to feature scale and initialization.
"""
        )

    m1, m2 = st.columns(2)
    with m1:
        with st.container(border=True):
            st.subheader("Centroids")
            st.markdown(
                """
A **centroid** is the center of a cluster. In standard K-Means, it is the **average
(mean)** position of all points currently assigned to that cluster.

If a cluster has points $(x_1,y_1), ..., (x_n,y_n)$, its centroid is:
"""
            )
            st.latex(r"C = \left(\frac{1}{n}\sum_{i=1}^{n} x_i,\; \frac{1}{n}\sum_{i=1}^{n} y_i\right)")
    with m2:
        with st.container(border=True):
            st.subheader("Euclidean distance")
            st.markdown(
                """
K-Means commonly uses **Euclidean distance** to measure how far a point is from a centroid.
For two 2D points $(x_1, y_1)$ and $(x_2, y_2)$:
"""
            )
            st.latex(r"d = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}")
            st.markdown(
                "Smaller distance means the point is closer, so it is more likely to join that cluster."
            )

    with st.expander("Inertia, Elbow Method, and Silhouette Score", expanded=False):
        st.markdown("**Inertia** (within-cluster sum of squares) measures cluster compactness:")
        st.latex(r"\text{Inertia} = \sum_{j=1}^{K}\sum_{x_i \in C_j} \lVert x_i - \mu_j \rVert^2")
        st.markdown(
            r"""
- $x_i$ is a data point, $\mu_j$ is the centroid of cluster $C_j$.
- Lower inertia means points are closer to their centroids.
- Inertia usually decreases as K increases, so lower is not always “better” by itself.

**Elbow Method**:
- Run K-Means for several K values (e.g., 2 to 10).
- Plot K vs inertia.
- Look for an “elbow” where inertia stops improving dramatically; that K is often a reasonable choice.

**Silhouette Score** (ranges roughly from -1 to 1):
- Compares how close a point is to its own cluster vs the nearest other cluster.
- Higher values mean better-separated, well-assigned clusters.
- Values near 0 suggest overlapping clusters; negative values suggest possible misassignment.
"""
        )

    with st.expander("Why StandardScaler matters", expanded=False):
        st.markdown(
            """
**StandardScaler** transforms each feature to have roughly mean 0 and standard deviation 1:

"""
        )
        st.latex(r"z = \frac{x - \mu}{\sigma}")
        st.markdown(
            """
Scaling matters for K-Means because it uses **distance**:
- Features with larger numeric ranges can dominate Euclidean distance.
- Example: income (0-100,000) can overpower age (0-100) unless scaled.
- Standardizing puts features on a comparable scale so clustering reflects all features more fairly.

Always fit the scaler on the training/data used for clustering preparation, and apply the same
transformation consistently.
"""
        )


def _render_interactive_demo() -> None:
    st.header("Interactive K-Means walkthrough")
    st.caption(
        "A small synthetic 2D example computed live with NumPy. Use the controls to step through iterations."
    )

    data = _make_demo_data(seed=7, n_per=5)

    controls, plot_col = st.columns([1, 2], gap="large")
    with controls:
        k = st.slider("Number of clusters (K)", min_value=2, max_value=4, value=3)
        init_seed = st.select_slider(
            "Initial centroid seed",
            options=[1, 3, 7, 11, 19],
            value=7,
            help="Changes which points are picked as starting centroids.",
        )
        max_iters = 10
        states = _run_kmeans_steps(data, k=k, init_seed=int(init_seed), max_iters=max_iters)
        step = st.slider(
            "Iteration step",
            min_value=0,
            max_value=len(states) - 1,
            value=0,
            help="Step 0 shows initial centroids. Later steps show assignment + centroid update.",
        )
        state = states[step]

        st.markdown("**What this step shows**")
        st.write(state["phase"])
        if state["labels"] is None:
            st.info(
                "No assignments yet. The algorithm has picked K starting centroids; next it will "
                "compute distances from every point to every centroid."
            )
        else:
            st.metric("Inertia (sum of squared distances)", f"{state['inertia']:.3f}")
            st.write(f"Max centroid shift: {state['max_centroid_shift']:.6f}")
            st.write("Converged:" + (" **Yes**" if state["converged"] else " No"))

    with plot_col:
        fig = _plot_state(data, state, k)
        st.plotly_chart(fig, width='stretch')

    st.subheader("Distance and assignment details")
    if state["labels"] is None or state["distances"] is None:
        st.markdown(
            """
At the next step, for each point the algorithm computes Euclidean distance to **every**
centroid, then assigns the point to the closest one.

Example (conceptual): if a point has distances `[2.1, 0.8, 3.4]` to centroids C1-C3, it joins
**C2** because 0.8 is smallest.
"""
        )
    else:
        labels = state["labels"]
        distances = state["distances"]
        rows = []
        for i in range(len(data)):
            row = {
                "Point": f"P{i + 1}",
                "x": round(float(data[i, 0]), 3),
                "y": round(float(data[i, 1]), 3),
                "Assigned": f"Cluster {int(labels[i]) + 1}",
            }
            for j in range(k):
                row[f"d to C{j + 1}"] = round(float(distances[i, j]), 3)
            rows.append(row)
        detail_df = pd.DataFrame(rows)
        st.dataframe(detail_df, width='stretch', hide_index=True)
        st.markdown(
            """
**How to read this table:** each row is a point. Distance columns show Euclidean distance to
each centroid. The assigned cluster is the centroid with the smallest distance in that row.
After assignment, K-Means recalculates each centroid as the mean of its assigned points.
"""
        )

    st.subheader("Convergence")
    if state["labels"] is None:
        st.write(
            "Move the iteration slider to see assignments and centroid updates. Convergence happens "
            "when centroids stop changing (within a small tolerance)."
        )
    elif state["converged"]:
        st.success(
            "The centroids stopped moving (within tolerance), so K-Means has converged for this demo."
        )
    else:
        st.info(
            "Not converged yet. Increase the iteration step if available; centroids may still move."
        )


def render() -> None:
    """Render the Learn K-Means educational module."""
    st.header("Learn K-Means")
    _render_concepts()
    st.divider()
    _render_interactive_demo()
    st.divider()
    st.caption(
        "Tip: after learning the concepts, try the Real-Life Example and K-Means Lab sections to "
        "run K-Means on real data with Scikit-learn."
    )
