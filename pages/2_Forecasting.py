from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Forecasting", layout="wide")

FINAL_CLUSTERS = list(range(8))


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def clean_cluster_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "cluster_id" not in df.columns:
        return df.copy()

    cleaned = df.copy()
    cleaned["cluster_id"] = pd.to_numeric(cleaned["cluster_id"], errors="coerce")
    cleaned = cleaned.dropna(subset=["cluster_id"])
    cleaned["cluster_id"] = cleaned["cluster_id"].astype(int)
    cleaned = cleaned[cleaned["cluster_id"].isin(FINAL_CLUSTERS)].copy()
    return cleaned


def load_label_map(labels_df: pd.DataFrame) -> dict[int, str]:
    if labels_df.empty or not {"cluster_id", "label"}.issubset(labels_df.columns):
        return {}

    labels = clean_cluster_df(labels_df)
    labels = labels.dropna(subset=["label"]).copy()
    labels["label"] = labels["label"].astype(str)

    return dict(zip(labels["cluster_id"], labels["label"]))


def format_cluster_name(cluster_id: int, label_map: dict[int, str]) -> str:
    return label_map.get(cluster_id, f"Cluster {cluster_id}")


forecasts_df = load_csv("data/processed/trends/cluster_forecasts.csv")
backtest_df = load_csv("data/processed/trends/forecast_backtest_metrics.csv")
labels_df = load_csv("data/processed/trends/cluster_labels.csv")

forecasts_df = clean_cluster_df(forecasts_df)
backtest_df = clean_cluster_df(backtest_df)
labels_df = clean_cluster_df(labels_df)

label_map = load_label_map(labels_df)

if not forecasts_df.empty:
    if "forecast_month" in forecasts_df.columns:
        forecasts_df["forecast_month"] = pd.to_datetime(
            forecasts_df["forecast_month"], errors="coerce"
        )

    if "label" not in forecasts_df.columns:
        forecasts_df["label"] = forecasts_df["cluster_id"].map(label_map)
    else:
        forecasts_df["label"] = forecasts_df["cluster_id"].map(label_map).fillna(
            forecasts_df["label"]
        )

    if "forecast_month" in forecasts_df.columns:
        forecasts_df = forecasts_df.sort_values(["cluster_id", "forecast_month"])

if not backtest_df.empty:
    if "label" not in backtest_df.columns:
        backtest_df["label"] = backtest_df["cluster_id"].map(label_map)
    else:
        backtest_df["label"] = backtest_df["cluster_id"].map(label_map).fillna(
            backtest_df["label"]
        )

    backtest_df = backtest_df.sort_values("cluster_id")


st.markdown(
    """
    <style>
    .hero-box {
        padding: 1.6rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #334155 100%);
        color: white;
        margin-bottom: 1.2rem;
    }

    .hero-box h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2.2rem;
    }

    .hero-box p {
        margin: 0;
        font-size: 1rem;
        line-height: 1.6;
        color: #dbeafe;
        max-width: 1000px;
    }

    .mini-card {
        background: #ffffff;
        padding: 1rem 1rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        min-height: 120px;
        color: #0f172a;  
    }

    .mini-card h4 {
        color: #0f172a;
        margin-bottom: 0.3rem;
    }

    .mini-card p {
        color: #334155;
        font-size: 0.9rem;
    }

    .small-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
    }

    .section-note {
        color: #475569;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-box">
        <h1>Forecasting</h1>
        <p>
            This page presents short-term forecast outputs for the final eight clusters and the backtest results used to assess forecast quality.
            The comparison is between a linear trend model and a naive persistence baseline, making it easier to judge whether the model adds value.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if forecasts_df.empty and backtest_df.empty:
    st.warning("Forecast and backtest files were not found, or no rows matched the final K=8 solution.")
    st.stop()

available_clusters = sorted(
    set(forecasts_df["cluster_id"].tolist() if not forecasts_df.empty else [])
    | set(backtest_df["cluster_id"].tolist() if not backtest_df.empty else [])
)

filter_col_1, filter_col_2 = st.columns([2, 1])

with filter_col_1:
    selected_cluster = st.selectbox(
        "Select a cluster to inspect",
        options=["All clusters"] + available_clusters,
        format_func=lambda x: x if x == "All clusters" else f"{x}: {format_cluster_name(x, label_map)}",
    )

with filter_col_2:
    st.markdown(
        """
        <div class="section-note" style="padding-top: 1.9rem;">
            Use the filter to switch from an overall view to a single-cluster demonstration.
        </div>
        """,
        unsafe_allow_html=True,
    )

display_forecasts = forecasts_df.copy()
display_backtest = backtest_df.copy()

if selected_cluster != "All clusters":
    display_forecasts = display_forecasts[display_forecasts["cluster_id"] == selected_cluster].copy()
    display_backtest = display_backtest[display_backtest["cluster_id"] == selected_cluster].copy()

total_clusters = (
    backtest_df["cluster_id"].nunique()
    if not backtest_df.empty
    else forecasts_df["cluster_id"].nunique()
)

linear_wins = (
    (backtest_df["better_model"] == "linear").sum()
    if not backtest_df.empty and "better_model" in backtest_df.columns
    else 0
)
naive_wins = (
    (backtest_df["better_model"] == "naive").sum()
    if not backtest_df.empty and "better_model" in backtest_df.columns
    else 0
)
ties = (
    (backtest_df["better_model"] == "tie").sum()
    if not backtest_df.empty and "better_model" in backtest_df.columns
    else 0
)

mean_mae_linear = (
    backtest_df["mae_linear"].mean()
    if not backtest_df.empty and "mae_linear" in backtest_df.columns
    else None
)
mean_mae_naive = (
    backtest_df["mae_naive"].mean()
    if not backtest_df.empty and "mae_naive" in backtest_df.columns
    else None
)

summary_1, summary_2, summary_3, summary_4, summary_5 = st.columns(5)
summary_1.metric("Final clusters", total_clusters)
summary_2.metric("Linear wins", linear_wins)
summary_3.metric("Naive wins", naive_wins)
summary_4.metric("Ties", ties)
summary_5.metric(
    "Avg MAE improvement",
    round(mean_mae_naive - mean_mae_linear, 4)
    if mean_mae_linear is not None and mean_mae_naive is not None
    else "N/A",
)

st.markdown("## Model Comparison")

comparison_col_1, comparison_col_2, comparison_col_3 = st.columns(3)

with comparison_col_1:
    st.markdown(
        """
        <div class="mini-card">
            <div class="small-label">Linear trend model</div>
            <h4>Projects recent movement</h4>
            <p>Uses the recent direction of cluster share to estimate the next short-term values.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with comparison_col_2:
    st.markdown(
        """
        <div class="mini-card">
            <div class="small-label">Naive baseline</div>
            <h4>Assumes persistence</h4>
            <p>Uses the most recent observed value as the next prediction, giving a simple benchmark.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with comparison_col_3:
    st.markdown(
        """
        <div class="mini-card">
            <div class="small-label">Interpretation</div>
            <h4>Relative performance matters</h4>
            <p>The key question is whether the linear model improves on a simple and defensible baseline.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not display_forecasts.empty and selected_cluster != "All clusters":
    st.markdown("## Forecast Trajectory")

    chart_df = display_forecasts.copy()

    if "forecast_month" in chart_df.columns and (
        "predicted_share_pct" in chart_df.columns or "predicted_share_pct_naive" in chart_df.columns
    ):
        fig, ax = plt.subplots(figsize=(9, 4.5))

        if "predicted_share_pct" in chart_df.columns:
            ax.plot(
                chart_df["forecast_month"],
                chart_df["predicted_share_pct"],
                marker="o",
                linewidth=2,
                label="Linear forecast",
            )

        if "predicted_share_pct_naive" in chart_df.columns:
            ax.plot(
                chart_df["forecast_month"],
                chart_df["predicted_share_pct_naive"],
                marker="o",
                linestyle="--",
                linewidth=2,
                label="Naive baseline",
            )

        ax.set_title(format_cluster_name(selected_cluster, label_map))
        ax.set_xlabel("Forecast month")
        ax.set_ylabel("Predicted share (%)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No chartable forecast columns were available for the selected cluster.")

st.markdown("## Forecast Table")
st.caption(
    """
This table shows the forecast outputs for the selected view.
"""
)

if not display_forecasts.empty:
    forecast_cols = [
        "cluster_id",
        "label",
        "forecast_month",
        "forecast_step",
        "predicted_share_pct",
        "predicted_share_pct_naive",
        "last_observed_share_pct",
        "slope",
        "intercept",
    ]
    forecast_cols = [col for col in forecast_cols if col in display_forecasts.columns]

    st.dataframe(
        display_forecasts[forecast_cols],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No forecast rows are available for the selected view.")

st.markdown("## Backtest Results")
st.caption(
    """
These results compare the linear model against the naive baseline on the final holdout point.
"""
)

if not display_backtest.empty:
    backtest_cols = [
        "cluster_id",
        "label",
        "actual_last_share_pct",
        "predicted_last_share_pct_linear",
        "predicted_last_share_pct_naive",
        "mae_linear",
        "mae_naive",
        "mape_pct_linear",
        "mape_pct_naive",
        "better_model",
        "mae_improvement_vs_naive",
    ]
    backtest_cols = [col for col in backtest_cols if col in display_backtest.columns]

    st.dataframe(
        display_backtest[backtest_cols],
        use_container_width=True,
        hide_index=True,
    )

    if "better_model" in display_backtest.columns and selected_cluster == "All clusters":
        st.markdown("### Clusters where the linear model beat the naive baseline")

        linear_better = display_backtest[
            display_backtest["better_model"] == "linear"
        ].copy()

        if not linear_better.empty:
            better_cols = [
                "cluster_id",
                "label",
                "mae_linear",
                "mae_naive",
                "mae_improvement_vs_naive",
            ]
            better_cols = [col for col in better_cols if col in linear_better.columns]

            st.dataframe(
                linear_better[better_cols],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("The linear model did not outperform the naive baseline on any cluster.")
else:
    st.info("No backtest rows are available for the selected view.")

with st.expander("How to interpret these results"):
    st.markdown(
        """
The forecast table gives a short-term estimate of where each cluster may move next.

The backtest table evaluates the forecasting logic on a final holdout point by comparing:
- a **linear trend model**
- a **naive persistence baseline**

This is a lightweight and exploratory evaluation rather than a production-grade forecasting framework.
Given the short and noisy time series, the main purpose is to test whether the linear model adds value beyond a simple baseline.
"""
    )
