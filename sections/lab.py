"""K-Means Lab section: run clustering on a user-uploaded CSV."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core import data_utils, kmeans_utils


def render() -> None:
    st.header("K-Means Lab")
    st.caption("Upload your own CSV dataset and run K-Means clustering.")

    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file to begin.")
        return

    try:
        df, info, err = data_utils.load_csv_safely(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load CSV: {exc}")
        return
    if err:
        st.error(err)
        return
    if df is None or df.empty:
        st.error("No data loaded.")
        return

    st.success(
        f"Loaded '{info['filename']}' — {info['row_count']} rows, {info['column_count']} columns."
    )

    st.subheader("First 10 rows")
    st.dataframe(df.head(10), width='stretch')

    st.subheader("Data types and missing values")
    st.dataframe(data_utils.get_missing_summary(df), width='stretch')
    st.write("Dtypes:")
    st.write(df.dtypes.astype(str))

    numeric_cols = data_utils.get_numeric_columns(df)
    if not numeric_cols:
        st.error("No numerical columns found. K-Means requires numeric features.")
        return

    st.subheader("Feature selection")
    features = st.multiselect(
        "Select clustering features (numeric only)", options=numeric_cols, default=numeric_cols
    )
    if not features:
        st.warning("Select at least one feature to continue.")
        return

    st.subheader("Missing value handling")
    strategy_label = st.selectbox(
        "Missing-value strategy",
        ["Mean imputation", "Median imputation", "Drop rows with missing values"],
    )
    strategy_map = {
        "Mean imputation": "mean",
        "Median imputation": "median",
        "Drop rows with missing values": "drop_rows",
    }
    strategy = strategy_map[strategy_label]

    try:
        work_df, dropped, miss_err = data_utils.handle_missing_values(df, features, strategy)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to handle missing values: {exc}")
        return
    if miss_err:
        st.error(miss_err)
        return
    st.info(f"Records after cleaning: {len(work_df)} (dropped {dropped}).")

    st.subheader("Scaling")
    scale_enabled = st.checkbox("Enable StandardScaler", value=True)

    try:
        X, scaler = data_utils.scale_features(work_df, enabled=scale_enabled)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to scale features: {exc}")
        return

    st.subheader("K-Means configuration")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        k = st.number_input("Number of clusters K", min_value=2, value=3, step=1)
    with col2:
        random_state = st.number_input("Random state", value=42, step=1)
    with col3:
        n_init = st.number_input("n_init", min_value=1, value=10, step=1)
    with col4:
        max_iter = st.number_input("Max iterations", min_value=1, value=300, step=1)

    rs = None if random_state is None else int(random_state)

    ok, verr = data_utils.validate_clustering_inputs(X, int(k), n_init=int(n_init), max_iter=int(max_iter), random_state=rs)
    if not ok:
        st.error(verr)
        return

    if st.button("Run K-Means"):
        try:
            result = kmeans_utils.fit_kmeans(
                X, int(k), random_state=rs, n_init=int(n_init), max_iter=int(max_iter)
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"K-Means training failed: {exc}")
            return

        labels = result["labels"]
        labeled = kmeans_utils.add_labels_to_copy(df, labels) if False else None
        # add labels to a copy of original dataset aligned by cleaned rows? Use cleaned index alignment
        # Build labeled copy of original df for rows that remain
        try:
            labeled_orig = df.copy()
            # align labels to original rows: only cleaned rows get labels; others NaN
            label_series = pd.Series(np.nan, index=df.index)
            # work_df index equals subset of original feature rows? handle_missing may drop rows
            # Reconstruct index mapping by re-running selection
            selected = df[features].copy()
            for c in features:
                selected[c] = pd.to_numeric(selected[c], errors="coerce")
            if strategy == "drop_rows":
                valid_idx = selected.dropna().index
            else:
                valid_idx = selected.index
            label_series.loc[valid_idx[: len(labels)]] = labels
            labeled_orig["Cluster"] = label_series
        except Exception:
            labeled_orig = kmeans_utils.add_labels_to_copy(df.copy(), labels) if len(labels) == len(df) else df.copy()

        st.session_state["lab_result"] = {
            "result": result,
            "work_df": work_df,
            "features": features,
            "scaler": scaler,
            "scale_enabled": scale_enabled,
            "labeled_orig": labeled_orig,
            "labels": labels,
        }

    if "lab_result" not in st.session_state:
        return

    R = st.session_state["lab_result"]
    result = R["result"]
    features = R["features"]
    work_df = R["work_df"]
    scaler = R["scaler"]
    scale_enabled = R["scale_enabled"]
    labels = R["labels"]

    st.subheader("Cluster counts")
    counts = kmeans_utils.cluster_counts(labels)
    st.dataframe(counts, width='stretch')

    st.subheader("Centroids (original feature units)")
    centroids = kmeans_utils.centroids_original_scale(
        result["centroids_scaled"], scaler, features, scale_enabled
    )
    st.dataframe(centroids, width='stretch')

    c1, c2 = st.columns(2)
    c1.metric("Inertia", f"{result['inertia']:.4f}")
    c2.metric("Iterations", result["n_iter"])

    # Scatter plots
    if len(features) >= 2:
        st.subheader("2D scatter plot")
        x_axis = st.selectbox("X axis", features, index=0, key="lab_x")
        y_axis = st.selectbox("Y axis", features, index=1 if len(features) > 1 else 0, key="lab_y")
        plot_df = work_df.copy()
        plot_df["Cluster"] = labels.astype(str)
        fig = px.scatter(plot_df, x=x_axis, y=y_axis, color="Cluster", title="Clusters")
        st.plotly_chart(fig, width='stretch')

    if len(features) >= 3:
        st.subheader("3D scatter plot")
        fig3 = px.scatter_3d(
            work_df.assign(Cluster=labels.astype(str)), x=features[0], y=features[1], z=features[2], color="Cluster"
        )
        st.plotly_chart(fig3, width='stretch')

    st.subheader("Cluster distribution")
    dist_fig = px.pie(counts, names="Cluster", values="Count", title="Cluster distribution")
    st.plotly_chart(dist_fig, width='stretch')

    # Downloads
    st.subheader("Downloads")
    labeled_download = kmeans_utils.add_labels_to_copy(work_df, labels)
    csv_bytes = labeled_download.to_csv(index=False).encode("utf-8")
    st.download_button("Download labeled data (CSV)", data=csv_bytes, file_name="labeled_data.csv", mime="text/csv")

    centroids_csv = centroids.to_csv(index=False).encode("utf-8")
    st.download_button("Download centroids (CSV)", data=centroids_csv, file_name="centroids.csv", mime="text/csv")

    counts_csv = counts.to_csv(index=False).encode("utf-8")
    st.download_button("Download cluster summary (CSV)", data=counts_csv, file_name="cluster_summary.csv", mime="text/csv")
