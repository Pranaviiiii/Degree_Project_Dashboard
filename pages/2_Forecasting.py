from __future__ import annotations

from pathlib import Path

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


st.title("Forecasting")
st.markdown(
    """
This page presents the short-term forecast outputs for the final eight clusters and the backtest results used to assess forecast quality.

The current approach compares:
- a **linear trend forecast**, which projects recent movement forward, and
- a **naive persistence baseline**, which assumes the next value will be the same as the most recent observed value.

This makes the forecast results easier to interpret, because model performance is judged against a simple baseline rather than in isolation.
"""
)

st.markdown("## Forecast Table")
st.caption(
    """
**Column guide**
- **cluster_id**: numeric cluster identifier  
- **label**: human-readable cluster name  
- **forecast_month**: month being predicted  
- **forecast_step**: how many steps ahead the prediction is  
- **predicted_share_pct**: forecast from the linear trend model  
- **predicted_share_pct_naive**: forecast from the naive persistence baseline  
- **last_observed_share_pct**: most recent observed cluster share used as the baseline reference  
- **slope**: trend direction and rate of change in the linear model  
- **intercept**: starting point of the linear model
"""
)

if not forecasts_df.empty:
    display_forecasts = forecasts_df.copy()

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
    st.warning("Forecast file not found, or no rows matched the final K=8 solution.")

st.markdown("## Forecast Backtest")
st.caption(
    """
**Column guide**
- **cluster_id**: numeric cluster identifier  
- **label**: cluster name  
- **actual_last_share_pct**: observed final share value  
- **predicted_last_share_pct_linear**: backtest prediction from the linear trend model  
- **predicted_last_share_pct_naive**: backtest prediction from the naive baseline  
- **mae_linear**: absolute error of the linear model, lower is better  
- **mae_naive**: absolute error of the naive baseline, lower is better  
- **mape_pct_linear**: percentage error of the linear model  
- **mape_pct_naive**: percentage error of the naive baseline  
- **better_model**: which model performed better on the final holdout point  
- **mae_improvement_vs_naive**: positive values mean the linear model improved on the naive baseline
"""
)

if not backtest_df.empty:
    display_backtest = backtest_df.copy()

    st.markdown("### Backtest Summary")

    total_clusters = display_backtest["cluster_id"].nunique()

    linear_wins = (
        (display_backtest["better_model"] == "linear").sum()
        if "better_model" in display_backtest.columns
        else 0
    )
    naive_wins = (
        (display_backtest["better_model"] == "naive").sum()
        if "better_model" in display_backtest.columns
        else 0
    )
    ties = (
        (display_backtest["better_model"] == "tie").sum()
        if "better_model" in display_backtest.columns
        else 0
    )

    mean_mae_linear = (
        display_backtest["mae_linear"].mean()
        if "mae_linear" in display_backtest.columns
        else None
    )
    mean_mae_naive = (
        display_backtest["mae_naive"].mean()
        if "mae_naive" in display_backtest.columns
        else None
    )

    metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
    metric_1.metric("Clusters tested", total_clusters)
    metric_2.metric("Linear wins", linear_wins)
    metric_3.metric("Naive wins", naive_wins)
    metric_4.metric("Ties", ties)
    metric_5.metric(
        "Avg MAE improvement",
        (
            round(mean_mae_naive - mean_mae_linear, 4)
            if mean_mae_linear is not None and mean_mae_naive is not None
            else "N/A"
        ),
    )

    st.markdown("### Detailed Backtest Results")

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

    if "better_model" in display_backtest.columns:
        st.markdown("### Clusters where the linear model beat the naive baseline")

        linear_better = display_backtest[
            display_backtest["better_model"] == "linear"
        ].copy()

        if not linear_better.empty:
            st.dataframe(
                linear_better[
                    [
                        col
                        for col in [
                            "cluster_id",
                            "label",
                            "mae_linear",
                            "mae_naive",
                            "mae_improvement_vs_naive",
                        ]
                        if col in linear_better.columns
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("The linear model did not outperform the naive baseline on any cluster.")
else:
    st.warning("Backtest metrics file not found, or no rows matched the final K=8 solution.")

with st.expander("How to interpret these results"):
    st.markdown(
        """
The forecast table gives a short-term estimate of where each cluster may move next.

The backtest table evaluates the forecasting logic on a final holdout point by comparing:
- a **linear trend model**, and
- a **naive persistence baseline**.

This is a lightweight and exploratory evaluation rather than a production-grade forecasting framework.  
Given the short and noisy time series, the main purpose is to test whether the linear model adds value beyond a simple baseline.
"""
    )

This is a lightweight and exploratory evaluation rather than a production-grade forecasting framework.  
Given the short and noisy time series, the main purpose is to assess whether the linear model adds value beyond a simple baseline.
"""
    )
