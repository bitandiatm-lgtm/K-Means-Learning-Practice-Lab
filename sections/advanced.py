"""Advanced Analysis section for the K-Means Streamlit app.

Self-contained, UI-only module that wires together core.data_utils,
core.kmeans_utils, core.metrics_utils, and core.sample_data to run an
interactive K-Means analysis workflow with Plotly visualizations and
CSV downloads.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data_utils import (
    get_numeric_columns,
    get_missing_summary,
    handle_missing_values,
    load_csv_safely,
    scale_features,
    validate_clustering_inputs,
)
from core.kmeans_utils import (
    add_labels_to_copy,
    centroids_original_scale,
    cluster_counts,
    cluster_feature_stats,
    fit_kmeans,
)
from core.metrics_utils import compute_inertia_range, compute_silhouette_range, recommend_k
from core.sample_data import get_sample_customer_df


# ---------------------------------------------------------------------------
# Small Plotly helpers
# ---------------------------------------------------------------------------
def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=13, color="#6b7280"),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        margin=dict(t=20, b=20, l=20, r=20),
    )
    return fig


def _line_chart(
    x: list[Any],
    y: list[float | None],
    title: str,
    x_title: str,
    y_title: str,
    marker_color: str = "#2563eb",
    highlight_x: int | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            marker=dict(color=marker_color, size=9),
            line=dict(color=marker_color, width=2),
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        )
    )
    if highlight_x is not None:
        fig.add_vline(
            x=highlight_x,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=f"Recommended K = {highlight_x}",
            annotation_position="top left",
        )
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
        template="simple_white",
        margin=dict(t=60, b=50, l=60, r=30),
    )
    return fig


def _bar_chart(
    x: list[Any],
    y: list[float | int],
    title: str,
    x_title: str,
    y_title: str,
    color: str = "#059669",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            marker_color=color,
            hovertemplate="%{x}: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="simple_white",
        margin=dict(t=60, b=50, l=60, r=30),
    )
    return fig


def _df_download_button(df: pd.DataFrame, label: str, file_name: str, key: str) -> None:
    if df is None or df.empty:
        return
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=file_name,
        mime="text/csv",
        key=key,
        width='stretch',
    )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
def render() -> None:
    st.header("Advanced Analysis")
    st.caption(
        "Run K-Means clustering and model-quality evaluation on a sample customer dataset "
        "or your own CSV. All results are computed live with Scikit-learn."
    )

    # ------------------------------------------------------------------
    # Data source
    # ------------------------------------------------------------------
    with st.expander("1. Choose data source", expanded=True):
        source = st.radio(
            "Data source",
            ["Sample customer dataset", "Upload CSV"],
            horizontal=True,
            key="adv_data_source",
        )
        df_raw: pd.DataFrame | None = None
        info: dict[str, Any] = {}
        err: str | None = None

        if source == "Sample customer dataset":
            try:
                df_raw = get_sample_customer_df()
                info = {
                    "filename": "sample_customers.csv (built-in)",
                    "row_count": int(df_raw.shape[0]),
                    "column_count": int(df_raw.shape[1]),
                }
            except Exception as exc:  # noqa: BLE001
                err = f"Failed to load sample dataset: {exc}"
        else:
            uploaded = st.file_uploader(
                "Upload a CSV file",
                type=["csv"],
                key="adv_csv_uploader",
                help="A customer-style CSV with numeric columns (e.g. Age, Income, Spending Score).",
            )
            if uploaded is not None:
                df_raw, info, err = load_csv_safely(uploaded)

        if err:
            st.error(err)
            st.stop()
        if df_raw is None or df_raw.empty:
            st.info("No data loaded yet. Select the sample dataset or upload a CSV.")
            st.stop()

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", info.get("row_count", df_raw.shape[0]))
        c2.metric("Columns", info.get("column_count", df_raw.shape[1]))
        c3.caption(f"Source: {info.get('filename', 'data')}")
        with st.expander("Preview data"):
            st.dataframe(df_raw.head(20), width='stretch', hide_index=True)

    # ------------------------------------------------------------------
    # Feature selection & preprocessing
    # ------------------------------------------------------------------
    with st.expander("2. Select features & preprocessing", expanded=True):
        numeric_cols = get_numeric_columns(df_raw)
        if not numeric_cols:
            st.error("No numeric columns found in the dataset. Cannot run K-Means.")
            st.stop()

        default_cols = [
            c for c in ("Age", "Annual Income", "Spending Score") if c in numeric_cols
        ] or numeric_cols[:3]
        features = st.multiselect(
            "Numeric features for clustering",
            numeric_cols,
            default=default_cols,
            key="adv_features",
        )
        if not features:
            st.warning("Select at least one numeric feature.")
            st.stop()

        missing_summary = get_missing_summary(df_raw[features])
        total_missing = int(missing_summary["missing_count"].sum()) if not missing_summary.empty else 0
        if total_missing > 0:
            st.info(f"Selected features contain {total_missing} missing value(s) in total.")
            with st.expander("Missing-value summary"):
                st.dataframe(missing_summary, width='stretch', hide_index=True)
            missing_strategy = st.selectbox(
                "Missing-value strategy",
                ["mean", "median", "drop_rows"],
                index=0,
                key="adv_missing_strategy",
                help="mean/median fill missing values per column; drop_rows removes rows with any missing value.",
            )
        else:
            missing_strategy = "mean"

        scale_enabled = st.toggle(
            "Standardize features (StandardScaler)",
            value=True,
            key="adv_scale",
            help="Recommended when features have different units/scales (e.g. Age vs Income).",
        )

        clean_df, dropped, clean_err = handle_missing_values(df_raw, features, missing_strategy)
        if clean_err:
            st.error(clean_err)
            st.stop()
        if dropped:
            st.caption(f"{dropped} row(s) removed/adjusted during missing-value handling.")

        try:
            X, scaler = scale_features(clean_df[features], enabled=scale_enabled)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        n_samples = int(X.shape[0])
        st.caption(f"Valid records for clustering: {n_samples} | Features: {', '.join(features)}")
        if n_samples < 2:
            st.error("Not enough valid records to cluster (need at least 2).")
            st.stop()

    # ------------------------------------------------------------------
    # K range configuration
    # ------------------------------------------------------------------
    with st.expander("3. Configure K range for evaluation", expanded=True):
        k_max_possible = max(2, n_samples)
        default_k_min = 2
        default_k_max = min(10, k_max_possible)
        if default_k_max < default_k_min:
            default_k_max = default_k_min

        col_k1, col_k2 = st.columns(2)
        k_min = col_k1.number_input(
            "Minimum K",
            min_value=2,
            max_value=k_max_possible,
            value=default_k_min,
            key="adv_k_min",
        )
        k_max = col_k2.number_input(
            "Maximum K",
            min_value=2,
            max_value=k_max_possible,
            value=default_k_max,
            key="adv_k_max",
        )
        if k_max < k_min:
            st.error("Maximum K must be >= Minimum K.")
            st.stop()
        # Silhouette needs k <= n_samples - 1; clamp note for the UI.
        silhouette_k_max = min(k_max, n_samples - 1)
        if silhouette_k_max < 2:
            st.info(
                "Dataset is too small for silhouette scores (need at least 3 records). "
                "Inertia/elbow analysis will still run."
            )

        run_eval = st.button("Run evaluation (inertia + silhouette)", key="adv_run_eval", type="primary")

    # ------------------------------------------------------------------
    # Evaluation: inertia + silhouette
    # ------------------------------------------------------------------
    inertia_results: list[dict[str, Any]] = []
    silhouette_results: list[dict[str, Any]] = []
    eval_df: pd.DataFrame | None = None

    if run_eval:
        with st.spinner("Computing inertia and silhouette scores..."):
            inertia_results = compute_inertia_range(
                X, k_min, k_max, random_state=42, n_init=10, max_iter=300
            )
            silhouette_results = compute_silhouette_range(
                X, k_min, silhouette_k_max, random_state=42, n_init=10, max_iter=300
            )

        # Build a combined evaluation table.
        rows: list[dict[str, Any]] = []
        for res in inertia_results:
            k = int(res["k"])
            sil = next((r["silhouette"] for r in silhouette_results if int(r["k"]) == k), None)
            rows.append(
                {"K": k, "Inertia": res.get("inertia"), "Silhouette": sil}
            )
        eval_df = pd.DataFrame(rows)
        st.session_state["adv_eval_df"] = eval_df
        st.session_state["adv_inertia_results"] = inertia_results
        st.session_state["adv_silhouette_results"] = silhouette_results

    # Restore from session if present (keeps results across widget interactions).
    if eval_df is None and "adv_eval_df" in st.session_state:
        eval_df = st.session_state["adv_eval_df"]
        inertia_results = st.session_state.get("adv_inertia_results", [])
        silhouette_results = st.session_state.get("adv_silhouette_results", [])

    if eval_df is not None and not eval_df.empty:
        st.subheader("Evaluation results")
        valid_inertia = eval_df[eval_df["Inertia"].notna()]
        valid_sil = eval_df[eval_df["Silhouette"].notna()]

        e1, e2, e3 = st.columns(3)
        if not valid_inertia.empty:
            best_elbow_k = int(valid_inertia.iloc[0]["K"])
            e1.metric("Lowest inertia (K)", best_elbow_k)
        else:
            e1.metric("Lowest inertia (K)", "—")
        if not valid_sil.empty:
            best_sil_row = valid_sil.loc[valid_sil["Silhouette"].idxmax()]
            e2.metric("Best silhouette (K)", int(best_sil_row["K"]))
            e3.metric("Best silhouette score", f"{float(best_sil_row['Silhouette']):.3f}")
        else:
            e2.metric("Best silhouette (K)", "—")
            e3.metric("Best silhouette score", "—")

        # Inertia (elbow) chart.
        inv_x = [int(r["K"]) for r in inertia_results if r.get("inertia") is not None]
        inv_y = [float(r["Inertia"]) for r in inertia_results if r.get("inertia") is not None]
        if inv_x:
            st.plotly_chart(
                _line_chart(
                    inv_x,
                    inv_y,
                    "Elbow Method: Inertia vs K",
                    "K (number of clusters)",
                    "Inertia (lower is better)",
                ),
                width='stretch',
            )
        else:
            st.plotly_chart(_empty_figure("No valid inertia results."), width='stretch')

        # Silhouette chart.
        sv_x = [int(r["K"]) for r in silhouette_results if r.get("silhouette") is not None]
        sv_y = [float(r["Silhouette"]) for r in silhouette_results if r.get("silhouette") is not None]
        if sv_x:
            best_sil_k = int(valid_sil.loc[valid_sil["Silhouette"].idxmax()]["K"]) if not valid_sil.empty else None
            st.plotly_chart(
                _line_chart(
                    sv_x,
                    sv_y,
                    "Silhouette Score vs K",
                    "K (number of clusters)",
                    "Silhouette score (higher is better, -1 to 1)",
                    marker_color="#059669",
                    highlight_x=best_sil_k,
                ),
                width='stretch',
            )
        else:
            st.plotly_chart(
                _empty_figure("No valid silhouette results (dataset may be too small)."),
                width='stretch',
            )

        with st.expander("Inertia / Silhouette table"):
            st.dataframe(eval_df, width='stretch', hide_index=True)
            _df_download_button(
                eval_df, "Download evaluation table (CSV)", "kmeans_evaluation.csv", "adv_dl_eval"
            )

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------
    st.subheader("Recommended K")
    rec_k: int | None = None
    rec_msg = ""
    if inertia_results or silhouette_results:
        rec_k, rec_msg = recommend_k(inertia_results, silhouette_results)
    st.info(rec_msg if rec_msg else "Run the evaluation above to get a K recommendation.")
    if rec_k is not None:
        st.caption(
            "This is a data-driven suggestion, not a definitive answer. Review the elbow and "
            "silhouette plots, and consider domain knowledge before choosing K."
        )

    # ------------------------------------------------------------------
    # Cluster analysis for a chosen K
    # ------------------------------------------------------------------
    st.subheader("Cluster analysis for a chosen K")
    k_choices = [int(r["K"]) for r in (eval_df["K"].tolist() if eval_df is not None else [])]
    if not k_choices:
        k_choices = list(range(int(k_min), int(k_max) + 1))

    default_k = rec_k if (rec_k in k_choices) else (k_choices[0] if k_choices else 2)
    chosen_k = st.selectbox(
        "Choose K to train and inspect",
        k_choices,
        index=k_choices.index(default_k) if default_k in k_choices else 0,
        key="adv_chosen_k",
    )

    train_btn = st.button("Train K-Means for chosen K", key="adv_train_btn", type="primary")

    if train_btn:
        ok, val_err = validate_clustering_inputs(X, chosen_k, random_state=42)
        if not ok:
            st.error(val_err)
            st.stop()
        result = fit_kmeans(X, chosen_k, random_state=42, n_init=10, max_iter=300)
        labels = result["labels"]
        centroids_scaled = result["centroids_scaled"]

        st.session_state["adv_labels"] = labels
        st.session_state["adv_chosen_k"] = int(chosen_k)
        st.session_state["adv_centroids_scaled"] = centroids_scaled
        st.session_state["adv_inertia"] = float(result["inertia"])
        st.session_state["adv_n_iter"] = int(result["n_iter"])

    labels = st.session_state.get("adv_labels")
    chosen_k = st.session_state.get("adv_chosen_k", chosen_k)
    centroids_scaled = st.session_state.get("adv_centroids_scaled")
    inertia_val = st.session_state.get("adv_inertia")
    n_iter_val = st.session_state.get("adv_n_iter")

    if labels is not None:
        ok, val_err = validate_clustering_inputs(X, chosen_k, random_state=42)
        if not ok:
            st.error(val_err)
            st.stop()

        m1, m2, m3 = st.columns(3)
        m1.metric("K", int(chosen_k))
        if inertia_val is not None:
            m2.metric("Inertia", f"{inertia_val:,.2f}")
        if n_iter_val is not None:
            m3.metric("Iterations to converge", int(n_iter_val))

        # Cluster distribution.
        counts_df = cluster_counts(labels)
        st.plotly_chart(
            _bar_chart(
                [f"Cluster {int(c)}" for c in counts_df["Cluster"]],
                counts_df["Count"].tolist(),
                "Cluster distribution (record counts)",
                "Cluster",
                "Count",
            ),
            width='stretch',
        )
        with st.expander("Cluster distribution table"):
            st.dataframe(counts_df, width='stretch', hide_index=True)
            _df_download_button(
                counts_df, "Download cluster counts (CSV)", "kmeans_cluster_counts.csv", "adv_dl_counts"
            )

        # Labeled data + centroids in original units.
        labeled_df = add_labels_to_copy(clean_df[features], labels, col_name="Cluster")
        centroids_df = centroids_original_scale(
            centroids_scaled, scaler, features, scaling_enabled=scale_enabled
        )

        st.subheader("Centroids (original feature units)")
        st.dataframe(centroids_df, width='stretch', hide_index=True)
        _df_download_button(
            centroids_df, "Download centroids (CSV)", "kmeans_centroids.csv", "adv_dl_centroids"
        )

        # Centroid visualization: grouped bar chart per feature.
        st.subheader("Centroid comparison")
        centroid_melted = centroids_df.melt(
            id_vars=["Cluster"], value_vars=features, var_name="Feature", value_name="Value"
        )
        fig = go.Figure()
        for feat in features:
            feat_df = centroid_melted[centroid_melted["Feature"] == feat]
            fig.add_trace(
                go.Bar(
                    name=feat,
                    x=[f"Cluster {int(c)}" for c in feat_df["Cluster"]],
                    y=feat_df["Value"],
                    hovertemplate="%{x}: %{y:.2f}<extra></extra>",
                )
            )
        fig.update_layout(
            title="Centroid values by feature (original units)",
            xaxis_title="Cluster",
            yaxis_title="Feature value",
            barmode="group",
            template="simple_white",
            hovermode="x unified",
            margin=dict(t=60, b=50, l=60, r=30),
        )
        st.plotly_chart(fig, width='stretch')

        # Cluster-wise mean and median.
        means_df, medians_df = cluster_feature_stats(labeled_df, features, label_col="Cluster")
        st.subheader("Cluster-wise feature statistics")
        tab_mean, tab_median = st.tabs(["Mean per cluster", "Median per cluster"])
        with tab_mean:
            st.dataframe(means_df, width='stretch', hide_index=True)
            _df_download_button(
                means_df, "Download cluster means (CSV)", "kmeans_cluster_means.csv", "adv_dl_means"
            )
        with tab_median:
            st.dataframe(medians_df, width='stretch', hide_index=True)
            _df_download_button(
                medians_df, "Download cluster medians (CSV)", "kmeans_cluster_medians.csv", "adv_dl_medians"
            )

        # Labeled data download.
        st.subheader("Labeled data (sample)")
        st.dataframe(labeled_df.head(50), width='stretch', hide_index=True)
        _df_download_button(
            labeled_df, "Download full labeled data (CSV)", "kmeans_labeled_data.csv", "adv_dl_labeled"
        )

        # Plain-language cluster descriptions.
        st.subheader("What the clusters look like")
        from core.kmeans_utils import describe_clusters

        descriptions = describe_clusters(centroids_df, features)
        if descriptions:
            for desc in descriptions:
                st.write("• " + desc)
        else:
            st.caption("Cluster descriptions unavailable.")

    # ------------------------------------------------------------------
    # Metric explanations
    # ------------------------------------------------------------------
    with st.expander("Metric explanations (plain language)", expanded=False):
        st.markdown(
            """
**Inertia (Within-Cluster Sum of Squares)**
- Measures how far points are from their assigned cluster center, summed over all clusters.
- Lower is better: tighter, more compact clusters.
- Caveat: inertia always decreases as K increases (with K = number of records, it reaches 0), so it should not be used alone to pick K.

**Elbow Method**
- Plot inertia against K and look for the "elbow" — the point where adding more clusters stops giving a big drop in inertia.
- Choosing K at the elbow balances compactness with simplicity. There is often no single sharp elbow; treat it as a hint.

**Silhouette Score**
- For each point, compares how close it is to its own cluster versus the next nearest cluster.
- Range: -1 to 1. Higher is better (well-separated, cohesive clusters). Near 0 means overlapping clusters; negative means points may be in the wrong cluster.
- Only defined for 2 <= K <= n_samples - 1 and requires at least 2 distinct clusters, so very small datasets may not produce a score.

**What "good" looks like**
- Low inertia *and* a relatively high silhouette score at the same K.
- A recommended K should also make sense for your data (e.g., known customer segments).

**Caveats**
- K-Means assumes roughly equal-sized, spherical clusters and is sensitive to outliers and feature scale.
- Results depend on the initial centroids; runs use multiple initializations (n_init=10) with a fixed random state for reproducibility.
- Standardizing features is usually important when variables use different units (e.g., Age vs Annual Income).
            """,
            unsafe_allow_html=False,
        )

