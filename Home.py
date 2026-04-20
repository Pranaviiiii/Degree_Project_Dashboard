# Home.py
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Home",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


def load_label_map(labels_df: pd.DataFrame) -> Dict[int, str]:
    if labels_df.empty or not {"cluster_id", "label"}.issubset(labels_df.columns):
        return {}

    labels = clean_cluster_df(labels_df)
    labels = labels.dropna(subset=["label"]).copy()
    labels["label"] = labels["label"].astype(str)

    return dict(zip(labels["cluster_id"], labels["label"]))


labels_df = load_csv("data/processed/trends/cluster_labels.csv")
trend_scores_df = load_csv("data/processed/trends/instagram_trend_scores.csv")
forecasts_df = load_csv("data/processed/trends/cluster_forecasts.csv")
backtest_df = load_csv("data/processed/trends/forecast_backtest_metrics.csv")
ig_exemplars_df = load_csv("data/processed/exemplars/instagram_exemplars.csv")
pin_exemplars_df = load_csv("data/processed/exemplars/pinterest_exemplars.csv")
trend_share_df = load_trend_share("data/processed/trends/instagram_cluster_share_pct.csv")

labels_df = clean_cluster_df(labels_df)
trend_scores_df = clean_cluster_df(trend_scores_df)
forecasts_df = clean_cluster_df(forecasts_df)
backtest_df = clean_cluster_df(backtest_df)
ig_exemplars_df = clean_cluster_df(ig_exemplars_df)
pin_exemplars_df = clean_cluster_df(pin_exemplars_df)

label_map = load_label_map(labels_df)

if not trend_scores_df.empty:
    if "label" not in trend_scores_df.columns:
        trend_scores_df["label"] = trend_scores_df["cluster_id"].map(label_map)
    else:
        trend_scores_df["label"] = trend_scores_df["cluster_id"].map(label_map).fillna(
            trend_scores_df["label"]
        )

if not forecasts_df.empty:
    if "label" not in forecasts_df.columns:
        forecasts_df["label"] = forecasts_df["cluster_id"].map(label_map)
    else:
        forecasts_df["label"] = forecasts_df["cluster_id"].map(label_map).fillna(
            forecasts_df["label"]
        )

if not backtest_df.empty:
    if "label" not in backtest_df.columns:
        backtest_df["label"] = backtest_df["cluster_id"].map(label_map)
    else:
        backtest_df["label"] = backtest_df["cluster_id"].map(label_map).fillna(
            backtest_df["label"]
        )


st.markdown(
    """
    <style>
    .hero {
        padding: 2.2rem 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #334155 100%);
        color: white;
        margin-bottom: 1.25rem;
    }

    .hero h1 {
        font-size: 2.5rem;
        margin-bottom: 0.4rem;
        color: #f8fafc;
        
    }

    .hero p {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #e2e8f0;
        max-width: 950px;
        
    }

    .info-card {
        background: #f8fafc;
        padding: 1rem 1.1rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        min-height: 165px;
        color: #1e293b;
    }

    .section-card {
        background: #ffffff;
        padding: 1.1rem 1.1rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
        color: #1e293b;
    }

    .info-card h4, .section-card h4 {
    color: #0f172a;
    margin-bottom: 0.4rem;
}

.info-card p, .section-card p {
    color: #334155;
    margin: 0;
}

    .small-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Fashion Trend Discovery and Forecasting Dashboard</h1>
        <p>
            This dashboard presents the final outputs of a multimodal fashion analysis pipeline
            built from Instagram captions and Pinterest images. The workflow uses CLIP embeddings,
            K-means clustering, monthly trend tracking, and lightweight forecasting to identify
            emerging and declining fashion content patterns. The final model uses an eight-cluster solution.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("## Project Overview")

overview_col_1, overview_col_2, overview_col_3 = st.columns(3)

with overview_col_1:
    st.markdown(
        """
        <div class="info-card">
            <div class="small-label">Data Sources</div>
            <h4>Instagram + Pinterest</h4>
            <p>
                Instagram captions provide the temporal trend signal, while Pinterest images add
                visual context and improve interpretation of each fashion cluster.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with overview_col_2:
    st.markdown(
        """
        <div class="info-card">
            <div class="small-label">Core Method</div>
            <h4>CLIP + K-means</h4>
            <p>
                Text and image inputs are embedded into a shared multimodal feature space and then
                grouped into recurring fashion archetypes using the final K=8 clustering solution.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with overview_col_3:
    st.markdown(
        """
        <div class="info-card">
            <div class="small-label">Dashboard Output</div>
            <h4>Trends + Forecasts</h4>
            <p>
                The dashboard tracks historical cluster share, highlights trend momentum, and compares
                simple forecast outputs against a naive baseline for short-term movement.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("## Key Findings")

if not trend_scores_df.empty:
    emerging_top = trend_scores_df.sort_values("momentum_score", ascending=False).head(1)
    declining_top = trend_scores_df.sort_values("momentum_score", ascending=True).head(1)
else:
    emerging_top = pd.DataFrame()
    declining_top = pd.DataFrame()

best_forecast_cluster = None
if not backtest_df.empty:
    if "mae_linear" in backtest_df.columns:
        best_forecast_cluster = backtest_df.sort_values("mae_linear", ascending=True).head(1)
    elif "mae" in backtest_df.columns:
        best_forecast_cluster = backtest_df.sort_values("mae", ascending=True).head(1)

date_range = "Unavailable"
if not trend_share_df.empty and len(trend_share_df.index) > 0:
    start_date = pd.to_datetime(trend_share_df.index.min()).strftime("%Y-%m")
    end_date = pd.to_datetime(trend_share_df.index.max()).strftime("%Y-%m")
    date_range = f"{start_date} to {end_date}"

finding_1, finding_2, finding_3, finding_4 = st.columns(4)

with finding_1:
    if not emerging_top.empty:
        st.info(f"**Top emerging cluster**\n\n{emerging_top.iloc[0]['label']}")
    else:
        st.info("**Top emerging cluster**\n\nUnavailable")

with finding_2:
    if not declining_top.empty:
        st.info(f"**Top declining cluster**\n\n{declining_top.iloc[0]['label']}")
    else:
        st.info("**Top declining cluster**\n\nUnavailable")

with finding_3:
    if best_forecast_cluster is not None and not best_forecast_cluster.empty:
        st.info(f"**Best backtest fit**\n\n{best_forecast_cluster.iloc[0]['label']}")
    else:
        st.info("**Best backtest fit**\n\nUnavailable")

with finding_4:
    st.info(f"**Date coverage**\n\n{date_range}")

st.markdown("## Pipeline Summary")

pipeline_col_1, pipeline_col_2, pipeline_col_3, pipeline_col_4 = st.columns(4)

with pipeline_col_1:
    st.markdown(
        """
        <div class="section-card">
            <div class="small-label">Step 1</div>
            <strong>Embed</strong>
            <p>Instagram captions and Pinterest images are encoded with CLIP into a shared semantic space.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with pipeline_col_2:
    st.markdown(
        """
        <div class="section-card">
            <div class="small-label">Step 2</div>
            <strong>Cluster</strong>
            <p>The combined embeddings are grouped into eight interpretable fashion content archetypes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with pipeline_col_3:
    st.markdown(
        """
        <div class="section-card">
            <div class="small-label">Step 3</div>
            <strong>Track</strong>
            <p>Monthly Instagram cluster shares are used to identify emerging, declining, and stable trends.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with pipeline_col_4:
    st.markdown(
        """
        <div class="section-card">
            <div class="small-label">Step 4</div>
            <strong>Forecast</strong>
            <p>A lightweight linear trend model is compared with a naive persistence baseline.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("## Overview Metrics")

metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)

n_clusters = len(label_map) if label_map else 0
n_trend_scores = len(trend_scores_df) if not trend_scores_df.empty else 0
n_forecasts = len(forecasts_df) if not forecasts_df.empty else 0
n_ig_examples = len(ig_exemplars_df) if not ig_exemplars_df.empty else 0
n_pin_examples = len(pin_exemplars_df) if not pin_exemplars_df.empty else 0

metric_1.metric("Final clusters", n_clusters)
metric_2.metric("Trend score rows", n_trend_scores)
metric_3.metric("Forecast rows", n_forecasts)
metric_4.metric("Instagram exemplars", n_ig_examples)
metric_5.metric("Pinterest exemplars", n_pin_examples)

st.markdown("## Platform Note")
st.info(
    "Pinterest content in this dataset is visually more consistent than Instagram and appears to contribute less cluster diversity. "
    "Instagram therefore drives most of the temporal variation used for trend scoring, while Pinterest mainly strengthens visual interpretation."
)

st.markdown("## How to Use the Dashboard")

nav_col_1, nav_col_2, nav_col_3 = st.columns(3)

with nav_col_1:
    st.markdown(
        """
        <div class="section-card">
            <strong>Trend Analysis</strong>
            <p>Review emerging and declining clusters and inspect how cluster share changes over time.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_col_2:
    st.markdown(
        """
        <div class="section-card">
            <strong>Forecasting</strong>
            <p>Compare forecast outputs and backtest metrics for the final eight-cluster solution.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_col_3:
    st.markdown(
        """
        <div class="section-card">
            <strong>Trend Explorer</strong>
            <p>Inspect a single cluster in more detail using charts, backtest results, and representative exemplars.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("Methodology and limitations"):
    st.markdown(
        """
### Method summary
- CLIP was used to embed Instagram captions and Pinterest images into a shared multimodal space
- K-means clustering grouped the combined embeddings into eight fashion archetypes
- Monthly Instagram cluster shares were used to compute momentum scores
- Short-term forecasts were generated using a simple linear trend model and compared with a naive baseline

### Important limitations
- The dataset is historical and covers a limited time window
- Forecasts are exploratory and based on a lightweight method
- Some clusters reflect commercial activity rather than purely stylistic trends
- Pinterest contributes less variation than Instagram in this dataset
- Interpretation is limited by the scope and structure of the available public data
"""
    )
