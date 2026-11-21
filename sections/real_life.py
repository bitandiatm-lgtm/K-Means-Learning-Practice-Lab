"""Real-Life Example: Customer Segmentation using K-Means.

All clustering and metrics are computed at runtime with Scikit-learn via core utils.
No hardcoded cluster labels or results are used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import data_utils, kmeans_utils, metrics_utils, sample_data


def _load_dataset() -> tuple[pd.DataFrame | None, str | None]:
    """Load the bundled sample dataset, creating the CSV if missing."""
    try:
        csv_path = sample_data.ensure_sample_csv()
        df, info, err = data_utils.load_csv_safely(csv_path)
        if err:
            return None, err
        return df, None
    except Exception as exc:  # noqa: BLE001
        return None, f"Failed to load sample dataset: {exc}"


def _plot_clusters_2d(
    plot_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    centroids_orig: pd.DataFrame,
) -> go.Figure:
    """Interactive 2D scatter with centroids."""
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=plot_df["Cluster"].astype(str),
        hover_data=[c for c in plot_df.columns if c != "Cluster"],
        title=f"Customer Segments: {x_col} vs {y_col}",
        labels={"color": "Cluster"},
        opacity=0.78,
    )
    fig.add_trace(
        go.Scatter(
            x=centroids_orig[x_col],
            y=centroids_orig[y_col],
            mode="markers+text",
            marker=dict(symbol="x", size=14, color="black", line=dict(width=2)),
            text=[f"C{int(c)}" for c in centroids_orig["Cluster"]],
            textposition="top center",
            name="Centroids",
            hovertemplate="<b>Centroid %{text}</b><br>"
            + f"{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra></extra>",
        )
    )
    fig.update_layout(legend_title_text="Cluster")
    return fig


def _plot_clusters_3d(
    plot_df: pd.DataFrame,
    cols: list[str],
    centroids_orig: pd.DataFrame,
) -> go.Figure:
    """Interactive 3D scatter with centroids."""
    x_col, y_col, z_col = cols[0], cols[1], cols[2]
    fig = px.scatter_3d(
        plot_df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=plot_df["Cluster"].astype(str),
        hover_data=[c for c in plot_df.columns if c != "Cluster"],
        title=f"Customer Segments (3D): {x_col}, {y_col}, {z_col}",
        labels={"color": "Cluster"},
        opacity=0.78,
    )
    fig.add_trace(
        go.Scatter3d(
            x=centroids_orig[x_col],
            y=centroids_orig[y_col],
            z=centroids_orig[z_col],
            mode="markers+text",
            marker=dict(symbol="x", size=7, color="black"),
            text=[f"C{int(c)}" for c in centroids_orig["Cluster"]],
            name="Centroids",
        )
    )
    fig.update_layout(legend_title_text="Cluster")
    return fig


def _line_metric_plot(results: list[dict[str, Any]], y_key: str, title: str, y_label: str) -> go.Figure:
    valid = [r for r in results if r.get(y_key) is not None]
    fig = go.Figure()
    if valid:
        ks = [int(r["k"]) for r in valid]
        ys = [float(r[y_key]) for r in valid]
        fig.add_trace(
            go.Scatter(
                x=ks,
                y=ys,
                mode="lines+markers",
                name=y_label,
                hovertemplate="K=%{x}<br>" + y_label + "=%{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Number of clusters (K)",
        yaxis_title=y_label,
        xaxis=dict(dtick=1),
        hovermode="x unified",
    )
    return fig


def render() -> None:
    st.header("Real-Life Example: Customer Segmentation")
    st.caption(
        "This workflow uses **K-Means** (from Scikit-learn) to group customers by similarity. "
        "All labels, centroids, and evaluation scores are computed from the data at runtime."
    )

    with st.expander("How K-Means works in practice (beginner guide)", expanded=False):
        st.markdown(
            """
1. **Collect features**: Choose numeric measurements that describe each customer (e.g., age, income, spending).
2. **Clean and prepare**: K-Means needs valid numbers. Missing values must be handled.
3. **Scale features**: K-Means uses distance. If one feature has a much larger range, it can dominate distance; standardization (mean 0, std 1) puts features on a comparable scale.
4. **Choose K**: K is the number of groups. The **Elbow Method** looks for where adding clusters gives smaller inertia improvement; the **Silhouette Score** measures how well-separated clusters are (higher is better, up to 1).
5. **Fit K-Means**: The algorithm assigns each point to the nearest centroid, recomputes centroids as cluster means, and repeats until assignments stabilize.
6. **Interpret clusters**: Compare centroids and cluster statistics in **original units** to describe each segment.
            """
        )

    df, err = _load_dataset()
    if err or df is None:
        st.error(err or "Unable to load the sample dataset.")
        return

    st.subheader("1) Sample customer dataset")
    st.write(
        "This synthetic customer dataset has no pre-assigned clusters. "
        "We will segment it using the features you select."
    )
    st.dataframe(df, width='stretch')
    st.caption(f"Records: {len(df)} | Columns: {len(df.columns)}")

    numeric_cols = data_utils.get_numeric_columns(df)
    default_features = [c for c in ["Age", "Annual Income", "Spending Score"] if c in numeric_cols]
    if not default_features:
        default_features = numeric_cols[: min(3, len(numeric_cols))]

    st.subheader("2) Select features and preprocessing")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        selected_features = st.multiselect(
            "Features for clustering",
            options=numeric_cols,
            default=default_features,
            help="Choose numeric columns. CustomerID is usually not a behavior feature; exclude it unless you have a specific reason.",
        )
    with col_b:
        scale_on = st.toggle(
            "Apply StandardScaler",
            value=True,
            help="Recommended when features have different units/ranges.",
        )
        missing_strategy = st.selectbox(
            "Missing-value handling",
            options=["median", "mean", "drop_rows"],
            index=0,
        )

    if "CustomerID" in selected_features:
        st.warning("CustomerID is selected. IDs are identifiers, not behavioral signals; clustering on IDs is usually not meaningful.")

    if len(selected_features) < 2:
        st.warning("Select at least 2 numeric features to visualize and cluster meaningfully.")
        return

    clean_df, dropped, clean_err = data_utils.handle_missing_values(df, selected_features, missing_strategy)
    if clean_err or clean_df is None:
        st.error(clean_err or "Could not prepare the selected features.")
        return
    if dropped:
        st.info(f"Dropped/adjusted rows during cleaning: {dropped}. Valid records: {len(clean_df)}.")

    n_records = len(clean_df)
    if n_records < 3:
        st.error("At least 3 valid records are required for clustering and silhouette evaluation.")
        return

    try:
        X, scaler = data_utils.scale_features(clean_df[selected_features], enabled=scale_on)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Scaling failed: {exc}")
        return

    st.subheader("3) Fit K-Means")
    st.write(
        "Pick **K** (number of clusters). Use the diagnostics below to compare K values; "
        "there is rarely one perfect answer—interpretability matters too."
    )
    k_max_ui = max(2, min(8, n_records))
    k_default = min(4, k_max_ui)
    ctrl1, ctrl2 = st.columns([1, 1])
    with ctrl1:
        k = st.slider("K (clusters)", min_value=2, max_value=k_max_ui, value=k_default, step=1)
    with ctrl2:
        st.caption("Model hyperparameters (advanced)")
    with st.expander("Advanced K-Means settings", expanded=False):
        random_state = st.number_input("random_state", min_value=0, max_value=9999, value=42, step=1)
        n_init = st.number_input("n_init (initializations)", min_value=1, max_value=50, value=10, step=1)
        max_iter = st.number_input("max_iter", min_value=50, max_value=1000, value=300, step=50)

    valid, val_err = data_utils.validate_clustering_inputs(
        X, int(k), int(n_init), int(max_iter), int(random_state)
    )
    if not valid:
        st.error(val_err or "Invalid clustering inputs.")
        return

    with st.spinner("Training K-Means..."):
        result = kmeans_utils.fit_kmeans(
            X,
            k=int(k),
            random_state=int(random_state),
            n_init=int(n_init),
            max_iter=int(max_iter),
        )

    labels = result["labels"]
    centroids_scaled = result["centroids_scaled"]
    centroids_orig = kmeans_utils.centroids_original_scale(
        centroids_scaled, scaler, selected_features, scale_on
    )
    counts = kmeans_utils.cluster_counts(labels)
    labeled_features = kmeans_utils.add_labels_to_copy(clean_df[selected_features], labels)
    labeled_full = kmeans_utils.add_labels_to_copy(clean_df, labels)

    m1, m2, m3 = st.columns(3)
    m1.metric("Inertia (WCSS)", f"{result['inertia']:.2f}")
    m2.metric("Iterations", f"{result['n_iter']}")
    m3.metric("Clusters", f"{int(result['k'])}")
    st.caption(
        "**Inertia** (within-cluster sum of squares) measures compactness in the scaled feature space. "
        "Lower is tighter, but inertia alone should not pick K because it usually decreases as K increases."
    )

    st.subheader("4) Cluster labels and centroids")
    st.write("Each row receives a cluster ID. The original feature table is not modified in place; labels are added to copies.")
    st.dataframe(labeled_full, width='stretch')

    st.markdown("**Centroids (cluster centers) in original units**")
    st.dataframe(centroids_orig, width='stretch')
    st.caption("Centroids are converted back to original units so you can interpret them as typical values per segment.")

    st.markdown("**Cluster distribution**")
    st.dataframe(counts, width='stretch')

    st.download_button(
        "Download labeled dataset (CSV)",
        data=labeled_full.to_csv(index=False).encode("utf-8"),
        file_name="customer_segments_labeled.csv",
        mime="text/csv",
    )

    st.subheader("5) Visualize clusters")
    st.write(
        "The scatter plot shows customers colored by their K-Means label. "
        "Black **X** markers are centroids. Points near the same centroid are more similar on the chosen axes."
    )
    plot_dims = st.radio("Plot type", options=["2D", "3D"], horizontal=True, index=0)
    if plot_dims == "3D":
        if len(selected_features) < 3:
            st.warning("3D plotting needs at least 3 selected features. Showing 2D instead.")
            x_col = st.selectbox("X axis", options=selected_features, index=0)
            y_opts = [c for c in selected_features if c != x_col] or selected_features
            y_col = st.selectbox("Y axis", options=y_opts, index=0)
            st.plotly_chart(_plot_clusters_2d(labeled_features, x_col, y_col, centroids_orig), width='stretch')
        else:
            d1, d2, d3 = st.columns(3)
            with d1:
                x_col = st.selectbox("X axis", options=selected_features, index=0)
            with d2:
                y_default = 1 if len(selected_features) > 1 else 0
                y_col = st.selectbox("Y axis", options=selected_features, index=y_default)
            with d3:
                z_default = 2 if len(selected_features) > 2 else 0
                z_col = st.selectbox("Z axis", options=selected_features, index=z_default)
            cols3 = [x_col, y_col, z_col]
            if len(set(cols3)) < 3:
                st.warning("Choose three different features for a 3D plot.")
            else:
                st.plotly_chart(_plot_clusters_3d(labeled_features, cols3, centroids_orig), width='stretch')
    else:
        p1, p2 = st.columns(2)
        with p1:
            x_col = st.selectbox("X axis", options=selected_features, index=0)
        with p2:
            y_default = 1 if len(selected_features) > 1 else 0
            y_col = st.selectbox("Y axis", options=selected_features, index=y_default)
        if x_col == y_col and len(selected_features) > 1:
            st.warning("X and Y are the same. Pick different features for a clearer view.")
        st.plotly_chart(_plot_clusters_2d(labeled_features, x_col, y_col, centroids_orig), width='stretch')

    st.subheader("6) Cluster characteristics (from data)")
    st.write(
        "These summaries are generated from the fitted centroids and cluster statistics—not hardcoded labels. "
        "Compare a cluster's values to the overall average to describe its behavior."
    )
    explanations = kmeans_utils.describe_clusters(centroids_orig, selected_features)
    for text in explanations:
        st.markdown(f"- {text}")

    means, medians = kmeans_utils.cluster_feature_stats(labeled_features, selected_features)
    stat_tab1, stat_tab2 = st.tabs(["Cluster means", "Cluster medians"])
    with stat_tab1:
        st.dataframe(means, width='stretch')
    with stat_tab2:
        st.dataframe(medians, width='stretch')

    st.subheader("7) Choose K with diagnostics")
    st.write(
        "The **Elbow Method** plots inertia across K values; look for a bend where gains slow down. "
        "The **Silhouette Score** ranges roughly from -1 to 1; values closer to 1 suggest better-separated clusters. "
        "Silhouette needs at least 2 clusters and fewer clusters than samples."
    )

    k_eval_max = max(2, min(10, n_records - 1))
    k_eval_min = 2
    if k_eval_max < k_eval_min:
        st.warning("Not enough records to compute K diagnostics across a range.")
        return
    eval_range = st.slider(
        "K range for evaluation",
        min_value=2,
        max_value=max(2, min(10, n_records)),
        value=(2, min(10, n_records)),
        step=1,
        help="Capped to valid sample sizes.",
    )
    e_min, e_max = int(eval_range[0]), int(eval_range[1])
    if e_min >= e_max:
        st.warning("Choose a range with at least two K values.")
        return

    inertia_results = metrics_utils.compute_inertia_range(
        X, e_min, e_max, random_state=int(random_state), n_init=int(n_init), max_iter=int(max_iter)
    )
    sil_results = metrics_utils.compute_silhouette_range(
        X, e_min, min(e_max, n_records - 1), random_state=int(random_state), n_init=int(n_init), max_iter=int(max_iter)
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            _line_metric_plot(inertia_results, "inertia", "Elbow Method (Inertia)", "Inertia"),
            width='stretch',
        )
    with c2:
        st.plotly_chart(
            _line_metric_plot(sil_results, "silhouette", "Silhouette Score by K", "Silhouette"),
            width='stretch',
        )

    with st.expander("Evaluation tables", expanded=False):
        st.dataframe(pd.DataFrame(inertia_results), width='stretch')
        st.dataframe(pd.DataFrame(sil_results), width='stretch')
        rec_k, rec_reason = metrics_utils.recommend_k(inertia_results, sil_results)
        if rec_k is not None:
            st.info(rec_reason)
        sil_valid = [r for r in sil_results if r.get("silhouette") is not None]
        if sil_valid:
            best = max(sil_valid, key=lambda r: r["silhouette"])
            st.success(
                f"Highest silhouette in this range: K={int(best['k'])} "
                f"(score={float(best['silhouette']):.4f}). This is a data-driven hint, not a mandatory choice."
            )
        errors = [r for r in inertia_results + sil_results if r.get("error")]
        if errors:
            st.warning(f"Some K values could not be evaluated ({len(errors)} issue(s)). Adjust the range or data.")

    st.caption(
        "Note: K-Means assumes roughly spherical clusters in the scaled space and can be sensitive to K, scaling, and initialization. "
        "Use these diagnostics together with domain understanding."
    )
