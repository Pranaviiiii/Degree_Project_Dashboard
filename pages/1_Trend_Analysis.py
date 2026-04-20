# pages/1_Trend_Analysis.py
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Trend Analysis", layout="wide")

FINAL_CLUSTERS = list(range(8))


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


@st.cache_data
def load_trend_share(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].copy()
    df = df.sort_index()

    valid_cols = []
    for col in df.columns:
        try:
            cid = int(str(col))
            if cid in FINAL_CLUSTERS:
                valid_cols.append(col)
        except Exception:
            continue

    return df[valid_cols].copy()


def clean_cluster_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "cluster_id" not in df.columns:
        return df.copy()

    cleaned = df.copy()
    cleaned["cluster_id"] = pd.to_numeric(cleaned["cluster_id"], errors="coerce")
    cleaned = cleaned.dropna(subset=["cluster_id"])
    cleaned["cluster_id"] = cleaned["cluster_id"].astype(int)
    cleaned = cleaned[cleaned["cluster_id"].isin(FINAL_CLUSTERS)].copy()
    return cleaned


def safe_int(value):
    try:
        return int(value)
    except Exception:
        return value


def get_cluster_type(label: str) -> str:
    label_l = str(label).lower()

    if any(
        word in label_l
        for word in [
            "resale",
            "promo",
            "promotion",
            "seller",
            "marketplace",
            "personal shopping",
            "preloved",
        ]
    ):
        return "Commerce-driven"

    if any(word in label_l for word in ["luxury", "boutique", "glam"]):
        return "Luxury"

    if any(word in label_l for word in ["blogger", "ootd", "lookbook"]):
        return "Editorial / Influencer"

    if any(word in label_l for word in ["menswear", "streetstyle", "denim"]):
        return "Style-driven"

    return "Mixed"


labels_df = load_csv("data/processed/trends/cluster_labels.csv")
trend_scores_df = load_csv("data/processed/trends/instagram_trend_scores.csv")
trend_share_df = load_trend_share("data/processed/trends/instagram_cluster_share_pct.csv")

labels_df = clean_cluster_df(labels_df)
trend_scores_df = clean_cluster_df(trend_scores_df)

label_map: Dict[int, str] = {}
if not labels_df.empty and {"cluster_id", "label"}.issubset(labels_df.columns):
    labels_df = labels_df.dropna(subset=["label"]).copy()
    labels_df["label"] = labels_df["label"].astype(str)
    label_map = dict(zip(labels_df["cluster_id"], labels_df["label"]))

if not trend_scores_df.empty:
    if "label" not in trend_scores_df.columns:
        trend_scores_df["label"] = trend_scores_df["cluster_id"].map(label_map)
    else:
        trend_scores_df["label"] = trend_scores_df["cluster_id"].map(label_map).fillna(
            trend_scores_df["label"]
        )

if not trend_share_df.empty:
    renamed_cols = {}
    for col in trend_share_df.columns:
        cid = safe_int(col)
        renamed_cols[col] = label_map.get(cid, f"Cluster {cid}")
    trend_share_named_df = trend_share_df.rename(columns=renamed_cols)
else:
    trend_share_named_df = pd.DataFrame()

st.title("Trend Analysis")
st.markdown(
    """
This page shows which clusters are currently emerging or declining and how their popularity changes over time.
The ranking is based on momentum scores derived from the monthly Instagram cluster share series for the final eight-cluster solution.
"""
)

max_clusters_available = max(3, len(FINAL_CLUSTERS))
default_top_n = min(5, max_clusters_available)

show_top_n = st.sidebar.slider(
    "Number of top clusters to display",
    min_value=3,
    max_value=max_clusters_available,
    value=default_top_n,
    key="trend_top_n",
    help="Controls how many of the strongest emerging and declining trends are shown in the tables and chart.",
)

st.sidebar.markdown(
    "Use this setting to focus on the most important changes in cluster momentum."
)

st.markdown("## Top Emerging and Declining Trends")
st.caption(
    "The number of clusters shown below is controlled by the sidebar slider."
)

if not trend_scores_df.empty:
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("### Emerging clusters")
        emerging = trend_scores_df.sort_values(
            "momentum_score", ascending=False
        ).head(show_top_n).copy()

        if not emerging.empty:
            emerging["cluster_type"] = emerging["label"].apply(get_cluster_type)

            emerging_cols = [
                "cluster_id",
                "label",
                "cluster_type",
                "momentum_score",
                "last_share_pct",
                "slope_3m_pct_points_per_month",
            ]
            emerging_cols = [col for col in emerging_cols if col in emerging.columns]

            st.dataframe(
                emerging[emerging_cols],
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Higher momentum scores indicate more strongly emerging clusters.")
        else:
            st.info("No emerging cluster data available.")

    with right_col:
        st.markdown("### Declining clusters")
        declining = trend_scores_df.sort_values(
            "momentum_score", ascending=True
        ).head(show_top_n).copy()

        if not declining.empty:
            declining["cluster_type"] = declining["label"].apply(get_cluster_type)

            declining_cols = [
                "cluster_id",
                "label",
                "cluster_type",
                "momentum_score",
                "last_share_pct",
                "slope_3m_pct_points_per_month",
            ]
            declining_cols = [col for col in declining_cols if col in declining.columns]

            st.dataframe(
                declining[declining_cols],
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Lower momentum scores indicate declining or unstable clusters.")
        else:
            st.info("No declining cluster data available.")
else:
    st.warning("Trend scores file not found, or no rows matched the final K=8 solution.")

st.markdown("## Trend Evolution")
st.caption(
    "This plot shows the historical monthly share of Instagram posts assigned to the most prominent final clusters."
)

if not trend_share_named_df.empty:
    fig, ax = plt.subplots(figsize=(12, 5))

    latest_sorted = trend_share_named_df.iloc[-1].sort_values(ascending=False)
    plot_cols = latest_sorted.head(show_top_n).index.tolist()

    for col in plot_cols:
        ax.plot(
            trend_share_named_df.index,
            trend_share_named_df[col],
            marker="o",
            linewidth=2,
            label=col,
        )

    ax.set_title("Monthly Share of Instagram Posts by Cluster")
    ax.set_xlabel("Month")
    ax.set_ylabel("Share of Posts (%)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
else:
    st.warning("Trend share file not found, or no final-cluster columns were available.")
