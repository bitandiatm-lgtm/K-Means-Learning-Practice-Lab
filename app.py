"""Main Streamlit app: K-Means Learning & Practice Lab."""

from __future__ import annotations

import streamlit as st

from core.sample_data import ensure_sample_csv
from sections import advanced, lab, learn, real_life


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        [data-testid="stSidebar"] { min-width: 280px; }
        .kpi-card {
            background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 12px;
            padding: 0.9rem 1rem; margin-bottom: 0.5rem;
        }
        .small-note { color: #475569; font-size: 0.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="K-Means Learning & Practice Lab", layout="wide")
    _inject_styles()
    ensure_sample_csv()

    st.title("K-Means Learning & Practice Lab")
    st.caption(
        "A professional, beginner-friendly dashboard to learn K-Means, explore a real-life example, "
        "run clustering on your own CSV, and analyze model quality with interactive visualizations."
    )

    with st.sidebar:
        st.header("Navigation")
        section = st.radio(
            "Go to",
            ["Learn K-Means", "Real-Life Example", "K-Means Lab", "Advanced Analysis"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown(
            "**Tech stack:** Python, Streamlit, Pandas, NumPy, Scikit-learn, Plotly, Matplotlib  \n"
            "**Note:** Clustering results are computed with Scikit-learn (no hardcoded labels)."
        )

    if section == "Learn K-Means":
        learn.render()
    elif section == "Real-Life Example":
        real_life.render()
    elif section == "K-Means Lab":
        lab.render()
    elif section == "Advanced Analysis":
        advanced.render()
    else:
        st.error("Unknown section.")


if __name__ == "__main__":
    main()
