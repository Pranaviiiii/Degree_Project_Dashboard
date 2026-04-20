from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Trend Explorer", layout="wide")

FINAL_CLUSTERS = list(range(8))


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


@st.cache_data
def load_share_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    return df


def coerce_cluster_ids(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "cluster_id" not in df.columns:
        return df.copy()

    clean_df = df.copy()
    clean_df["cluster_id"] = pd.to_numeric(clean_df["cluster_id"], errors="coerce")
    clean_df = clean_df.dropna(subset=["cluster_id"])
    clean_df["cluster_id"] = clean_df["cluster_id"].astype(int)
    clean_df = clean_df[clean_df["cluster_id"].isin(FINAL_CLUSTERS)].copy()
    return clean_df


def load_label_map(labels_df: pd.DataFrame) -> Dict[int, str]:
    if labels_df.empty or not {"cluster_id", "label"}.issubset(labels_df.columns):
        return {}

    clean_labels = coerce_cluster_ids(labels_df)
    clean_labels = clean_labels.dropna(subset=["label"]).copy()
    clean_labels["label"] = clean_labels["label"].astype(str)

    return dict(zip(clean_labels["cluster_id"], clean_labels["label"]))


def get_cluster_type(label: str) -> str:
    label_lower = str(label).lower()

    if any(
        token in label_lower
        for token in [
            "promo",
            "promotion",
            "shop",
            "seller",
            "resale",
            "preloved",
            "marketplace",
        ]
    ):
        return "Commerce-driven"

    if any(
        token in label_lower
        for token in ["luxury", "boutique", "designer", "wedding", "glam"]
    ):
        return "Luxury"

    if any(
        token in label_lower
        for token in ["blogger", "ootd", "streetstyle", "streetwear", "fashion week"]
    ):
        return "Editorial / Influencer"

    return "Style-driven"


def cluster_summary_sentence(label: str, latest_share: float, momentum: float) -> str:
    if momentum > 8:
        direction = "showing strong upward momentum"
    elif momentum > 2:
        direction = "showing moderate upward momentum"
    elif momentum < -8:
        direction = "declining sharply"
    elif momentum < -2:
        direction = "softening"
    else:
        direction = "remaining broadly stable"

    return (
        f"'{label}' currently accounts for {latest_share:.2f}% of observed Instagram cluster share "
        f"and is {direction}."
    )


def to_long_share_df(share_df: pd.DataFrame) -> pd.DataFrame:
    if share_df.empty:
        return pd.DataFrame(columns=["month", "cluster_id", "share_pct"])

    df = share_df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()

    valid_cluster_columns: List[str] = []
    for col in df.columns:
        try:
            cluster_id = int(str(col))
            if cluster_id in FINAL_CLUSTERS:
                valid_cluster_columns.append(col)
        except Exception:
            continue

    if not valid_cluster_columns:
        return pd.DataFrame(columns=["month", "cluster_id", "share_pct"])

    df = df[valid_cluster_columns].copy()
    df["month"] = df.index

    long_df = df.melt(
        id_vars=["month"],
        var_name="cluster_id",
        value_name="share_pct",
    )

    long_df["cluster_id"] = pd.to_numeric(long_df["cluster_id"], errors="coerce")
    long_df["share_pct"] = pd.to_numeric(long_df["share_pct"], errors="coerce")
    long_df = long_df.dropna(subset=["cluster_id", "share_pct"]).copy()
    long_df["cluster_id"] = long_df["cluster_id"].astype(int)
    long_df = long_df[long_df["cluster_id"].isin(FINAL_CLUSTERS)].copy()

    return long_df.sort_values(["cluster_id", "month"])


def build_cluster_options(
    share_long: pd.DataFrame,
    labels_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    instagram_ex_df: pd.DataFrame,
    pinterest_ex_df: pd.DataFrame,
) -> List[int]:
    cluster_ids = set()

    for df in [
        share_long,
        labels_df,
        forecast_df,
        backtest_df,
        instagram_ex_df,
        pinterest_ex_df,
    ]:
        if not df.empty and "cluster_id" in df.columns:
            cluster_ids.update(df["cluster_id"].astype(int).tolist())

    cluster_ids = {cluster_id for cluster_id in cluster_ids if cluster_id in FINAL_CLUSTERS}
    return sorted(cluster_ids)


def safe_float(row: pd.Series, column: str) -> float:
    if column in row.index and pd.notna(row[column]):
        return float(row[column])
    return float("nan")


def resolve_local_image_path(raw_path: object) -> str | None:
    if pd.isna(raw_path):
        return None

    raw_value = str(raw_path).strip()
    if not raw_value:
        return None

    if raw_value.startswith("http://") or raw_value.startswith("https://"):
        return None

    direct_path = Path(raw_value)
    if direct_path.exists():
        return str(direct_path)

    cwd_path = Path.cwd() / raw_value
    if cwd_path.exists():
        return str(cwd_path)

    data_path = Path("data") / raw_value
    if data_path.exists():
        return str(data_path)

    return None


def first_existing_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


share_df = load_share_csv("data/processed/trends/instagram_cluster_share_pct.csv")
labels_df = load_csv("data/processed/trends/cluster_labels.csv")
forecast_df = load_csv("data/processed/trends/cluster_forecasts.csv")
backtest_df = load_csv("data/processed/trends/forecast_backtest_metrics.csv")
instagram_ex_df = load_csv("data/processed/exemplars/instagram_exemplars.csv")
pinterest_ex_df = load_csv("data/processed/exemplars/pinterest_exemplars.csv")

share_long = to_long_share_df(share_df)
labels_df = coerce_cluster_ids(labels_df)
forecast_df = coerce_cluster_ids(forecast_df)
backtest_df = coerce_cluster_ids(backtest_df)
instagram_ex_df = coerce_cluster_ids(instagram_ex_df)
pinterest_ex_df = coerce_cluster_ids(pinterest_ex_df)

if not forecast_df.empty and "forecast_month" in forecast_df.columns:
    forecast_df["forecast_month"] = pd.to_datetime(forecast_df["forecast_month"], errors="coerce")
    forecast_df = forecast_df.dropna(subset=["forecast_month"]).sort_values("forecast_month")

label_map = load_label_map(labels_df)

cluster_ids = build_cluster_options(
    share_long,
    labels_df,
    forecast_df,
    backtest_df,
    instagram_ex_df,
    pinterest_ex_df,
)

st.title("Trend Explorer")

if not cluster_ids:
    st.warning("No trend data is available for the final eight-cluster solution.")
    st.stop()

display_names = {
    cluster_id: label_map.get(cluster_id, f"Cluster {cluster_id}")
    for cluster_id in cluster_ids
}

selected_cluster_id = st.sidebar.selectbox(
    "Choose a trend",
    options=cluster_ids,
    format_func=lambda cluster_id: display_names[cluster_id],
)

selected_label = label_map.get(selected_cluster_id, f"Cluster {selected_cluster_id}")

cluster_share = (
    share_long[share_long["cluster_id"] == selected_cluster_id]
    .copy()
    .sort_values("month")
)

cluster_forecast = (
    forecast_df[forecast_df["cluster_id"] == selected_cluster_id]
    .copy()
    .sort_values("forecast_month")
    if not forecast_df.empty
    else pd.DataFrame()
)

cluster_backtest = (
    backtest_df[backtest_df["cluster_id"] == selected_cluster_id].copy()
    if not backtest_df.empty
    else pd.DataFrame()
)

cluster_instagram_ex = (
    instagram_ex_df[instagram_ex_df["cluster_id"] == selected_cluster_id].copy()
    if not instagram_ex_df.empty
    else pd.DataFrame()
)

cluster_pinterest_ex = (
    pinterest_ex_df[pinterest_ex_df["cluster_id"] == selected_cluster_id].copy()
    if not pinterest_ex_df.empty
    else pd.DataFrame()
)

latest_share = float(cluster_share["share_pct"].iloc[-1]) if not cluster_share.empty else 0.0

momentum_score = 0.0
if len(cluster_share) >= 2:
    reference_index = max(0, len(cluster_share) - 3)
    momentum_score = float(
        cluster_share["share_pct"].iloc[-1] - cluster_share["share_pct"].iloc[reference_index]
    )

trend_type = get_cluster_type(selected_label)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Trend ID", selected_cluster_id)
metric_2.metric("Latest Share", f"{latest_share:.2f}%")
metric_3.metric("Momentum Score", f"{momentum_score:.2f}")
metric_4.metric("Trend Type", trend_type)

st.markdown(f"**Label:** {selected_label}")
st.caption(cluster_summary_sentence(selected_label, latest_share, momentum_score))

tab_charts, tab_backtest, tab_exemplars = st.tabs(["Charts", "Backtest", "Exemplars"])

with tab_charts:
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("### Historical Trend")

        if cluster_share.empty:
            st.info("No historical trend data is available for this cluster.")
        else:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(
                cluster_share["month"],
                cluster_share["share_pct"],
                marker="o",
                linewidth=2,
            )
            ax.set_title(selected_label)
            ax.set_xlabel("Month")
            ax.set_ylabel("Cluster Share of Posts (%)")
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

    with right_col:
        st.markdown("### Forecast View")

        if cluster_share.empty and cluster_forecast.empty:
            st.info("No forecast data is available for this cluster.")
        else:
            fig, ax = plt.subplots(figsize=(7, 4))

            if not cluster_share.empty:
                ax.plot(
                    cluster_share["month"],
                    cluster_share["share_pct"],
                    marker="o",
                    linewidth=2,
                    label="Historical share",
                )

            if not cluster_forecast.empty and "predicted_share_pct" in cluster_forecast.columns:
                ax.plot(
                    cluster_forecast["forecast_month"],
                    cluster_forecast["predicted_share_pct"],
                    marker="o",
                    linestyle="--",
                    linewidth=2,
                    label="Linear forecast",
                )

            if not cluster_forecast.empty and "predicted_share_pct_naive" in cluster_forecast.columns:
                ax.plot(
                    cluster_forecast["forecast_month"],
                    cluster_forecast["predicted_share_pct_naive"],
                    marker="o",
                    linestyle=":",
                    linewidth=2,
                    label="Naive baseline",
                )

            ax.set_title(selected_label)
            ax.set_xlabel("Month")
            ax.set_ylabel("Cluster Share of Posts (%)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

with tab_backtest:
    st.markdown("### Forecast Backtest")

    if cluster_backtest.empty:
        st.info("No backtest metrics are available for this cluster.")
    else:
        row = cluster_backtest.iloc[0]

        actual_last_share = safe_float(row, "actual_last_share_pct")
        predicted_linear = (
            safe_float(row, "predicted_last_share_pct_linear")
            if "predicted_last_share_pct_linear" in row.index
            else safe_float(row, "predicted_last_share_pct")
        )
        predicted_naive = safe_float(row, "predicted_last_share_pct_naive")

        mae_linear = (
            safe_float(row, "mae_linear")
            if "mae_linear" in row.index
            else safe_float(row, "mae")
        )
        mae_naive = safe_float(row, "mae_naive")

        mape_linear = (
            safe_float(row, "mape_pct_linear")
            if "mape_pct_linear" in row.index
            else safe_float(row, "mape_pct")
        )
        mape_naive = safe_float(row, "mape_pct_naive")

        better_model = (
            row["better_model"]
            if "better_model" in row.index and pd.notna(row["better_model"])
            else "N/A"
        )

        improvement = safe_float(row, "mae_improvement_vs_naive")

        backtest_metric_1, backtest_metric_2, backtest_metric_3 = st.columns(3)
        backtest_metric_1.metric("Actual last share", f"{actual_last_share:.2f}%")
        backtest_metric_2.metric("Linear MAE", f"{mae_linear:.2f}")
        backtest_metric_3.metric(
            "Naive MAE",
            f"{mae_naive:.2f}" if pd.notna(mae_naive) else "N/A",
        )

        detail_left, detail_right = st.columns(2)

        with detail_left:
            st.markdown(f"**Linear forecast:** {predicted_linear:.2f}%")
            st.markdown(f"**Linear MAPE:** {mape_linear:.2f}%")

        with detail_right:
            if pd.notna(predicted_naive):
                st.markdown(f"**Naive baseline forecast:** {predicted_naive:.2f}%")
            if pd.notna(mape_naive):
                st.markdown(f"**Naive MAPE:** {mape_naive:.2f}%")

        if better_model != "N/A":
            st.markdown(f"**Better model:** {better_model}")

        if pd.notna(improvement):
            if improvement > 0:
                st.success(
                    f"Linear model improved on the naive baseline by {improvement:.2f} MAE points."
                )
            elif improvement < 0:
                st.info(
                    f"Naive baseline outperformed the linear model by {abs(improvement):.2f} MAE points."
                )
            else:
                st.info("Linear and naive models performed equally on this cluster.")

with tab_exemplars:
    instagram_col, pinterest_col = st.columns(2)

    with instagram_col:
        st.markdown("### Instagram Exemplars")

        if cluster_instagram_ex.empty:
            st.info("No Instagram exemplars are available.")
        else:
            text_col = first_existing_column(
                cluster_instagram_ex,
                ["caption", "text", "post_text", "caption_clean", "clean_caption", "content"],
            )
            distance_col = first_existing_column(
                cluster_instagram_ex,
                ["distance_to_centroid", "distance", "centroid_distance", "score"],
            )

            display_cols: List[str] = [col for col in ["rank"] if col in cluster_instagram_ex.columns]
            if text_col:
                display_cols.append(text_col)
            if distance_col:
                display_cols.append(distance_col)

            if len(display_cols) <= 1:
                display_cols = [col for col in cluster_instagram_ex.columns if col != "cluster_id"][:4]

            display_df = cluster_instagram_ex[display_cols].copy()

            rename_map = {}
            if text_col and text_col in display_df.columns:
                rename_map[text_col] = "caption"
            if distance_col and distance_col in display_df.columns:
                rename_map[distance_col] = "distance_to_centroid"

            display_df = display_df.rename(columns=rename_map)

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            with st.expander("Show raw Instagram exemplar columns"):
                st.write(list(cluster_instagram_ex.columns))

    with pinterest_col:
        st.markdown("### Pinterest Exemplars")

        if cluster_pinterest_ex.empty:
            st.info("No Pinterest exemplars are available.")
        else:
            text_col = first_existing_column(
                cluster_pinterest_ex,
                ["caption", "text", "post_text", "title", "description", "content"],
            )
            distance_col = first_existing_column(
                cluster_pinterest_ex,
                ["distance_to_centroid", "distance", "centroid_distance", "score"],
            )
            image_col = first_existing_column(
                cluster_pinterest_ex,
                ["image_path", "image_url", "url", "img_url", "image"],
            )

            preview_df = cluster_pinterest_ex.copy()

            if image_col:
                preview_df["resolved_image_path"] = preview_df[image_col].apply(resolve_local_image_path)
            else:
                preview_df["resolved_image_path"] = None

            image_examples = preview_df[preview_df["resolved_image_path"].notna()].head(6).copy()

            if not image_examples.empty:
                image_grid_cols = st.columns(2)
                for i, (_, example_row) in enumerate(image_examples.iterrows()):
                    with image_grid_cols[i % 2]:
                        st.image(example_row["resolved_image_path"], use_container_width=True)

                        if text_col and pd.notna(example_row.get(text_col)):
                            st.caption(str(example_row[text_col])[:160])

                        if distance_col and pd.notna(example_row.get(distance_col)):
                            st.caption(
                                f"Distance to centroid: {float(example_row[distance_col]):.4f}"
                            )
            else:
                st.info(
                    "Pinterest exemplars are available, but no valid local image files were found from the stored paths."
                )

            display_cols: List[str] = [col for col in ["rank"] if col in cluster_pinterest_ex.columns]
            if text_col:
                display_cols.append(text_col)
            if image_col:
                display_cols.append(image_col)
            if distance_col:
                display_cols.append(distance_col)

            if len(display_cols) <= 1:
                display_cols = [col for col in cluster_pinterest_ex.columns if col != "cluster_id"][:5]

            display_df = cluster_pinterest_ex[display_cols].copy()

            rename_map = {}
            if text_col and text_col in display_df.columns:
                rename_map[text_col] = "caption"
            if image_col and image_col in display_df.columns:
                rename_map[image_col] = "image_path"
            if distance_col and distance_col in display_df.columns:
                rename_map[distance_col] = "distance_to_centroid"

            display_df = display_df.rename(columns=rename_map)

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            with st.expander("Show raw Pinterest exemplar columns"):
                st.write(list(cluster_pinterest_ex.columns))

with st.expander("How to interpret this page"):
    st.markdown(
        """
This page brings together the historical trend profile, short-term forecast, backtest evidence, and representative exemplars for one cluster.

The forecast view compares a linear trend model with a naive persistence baseline.
Because the time series is short and noisy, the forecasting results should be treated as exploratory rather than definitive.

The exemplar section helps interpret the content of each cluster by showing representative Instagram captions and Pinterest examples closest to the centroid.
"""
    )
