from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# set matplotlib env before importing pyplot
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


BASE_DIR = Path("/Users/andy/thesis")
RESULT_XLSX = BASE_DIR / "ballet_lda_results.xlsx"
OUT_DIR = BASE_DIR / "topic_trend_graphs"
OUT_DIR.mkdir(exist_ok=True)
OUT_XLSX = BASE_DIR / "ballet_topic_trend_tables.xlsx"

PERIOD_ORDER = ["1기(~1995)", "2기(1996~2005)", "3기(2006~2015)", "4기(2016~2025)"]


def set_font() -> None:
    candidates = [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    selected = None
    for p in candidates:
        if Path(p).exists():
            try:
                fm.fontManager.addfont(p)
                selected = fm.FontProperties(fname=p).get_name()
                break
            except Exception:
                continue
    if selected:
        plt.rcParams["font.family"] = [selected, "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    if not RESULT_XLSX.exists():
        raise FileNotFoundError(RESULT_XLSX)

    topics_df = pd.read_excel(RESULT_XLSX, sheet_name="topics_by_period")
    doc_df = pd.read_excel(RESULT_XLSX, sheet_name="doc_dominant_topic")

    # dominant topic -> sync topic mapping
    topic_map = topics_df[["period", "topic_id", "sync_topic"]].copy()
    topic_map["topic_id"] = topic_map["topic_id"].astype(int)
    doc_df["dominant_topic"] = doc_df["dominant_topic"].astype(int)

    merged = doc_df.merge(
        topic_map,
        left_on=["period", "dominant_topic"],
        right_on=["period", "topic_id"],
        how="left",
    )

    # count trend
    count_tbl = (
        merged.groupby(["sync_topic", "period"], as_index=False)
        .size()
        .rename(columns={"size": "건수"})
        .pivot(index="sync_topic", columns="period", values="건수")
        .fillna(0)
        .reindex(columns=PERIOD_ORDER)
        .reset_index()
    )

    # ratio trend (%)
    ratio_src = topics_df.groupby(["sync_topic", "period"], as_index=False)["topic_share_percent"].sum()
    ratio_tbl = (
        ratio_src.pivot(index="sync_topic", columns="period", values="topic_share_percent")
        .fillna(0.0)
        .reindex(columns=PERIOD_ORDER)
        .reset_index()
    )

    # long for plotting
    count_long = count_tbl.melt(id_vars="sync_topic", var_name="period", value_name="건수")
    ratio_long = ratio_tbl.melt(id_vars="sync_topic", var_name="period", value_name="비율(%)")

    # filter to visible lines only (at least one non-zero)
    active_topics = set(count_long.groupby("sync_topic")["건수"].sum().loc[lambda s: s > 0].index)
    count_long = count_long[count_long["sync_topic"].isin(active_topics)]
    ratio_long = ratio_long[ratio_long["sync_topic"].isin(active_topics)]

    set_font()

    # graph 1: count trend
    fig, ax = plt.subplots(figsize=(13, 8))
    for topic, sub in count_long.groupby("sync_topic"):
        sub = sub.set_index("period").reindex(PERIOD_ORDER).reset_index()
        ax.plot(sub["period"], sub["건수"], marker="o", linewidth=2, label=topic)
    ax.set_title("발레 담론 토픽 건수 변화", fontsize=22, pad=14)
    ax.set_xlabel("시기")
    ax.set_ylabel("건수")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    plt.tight_layout()
    count_png = OUT_DIR / "topic_count_trend.png"
    plt.savefig(count_png, dpi=220)
    plt.close(fig)

    # graph 2: ratio trend
    fig, ax = plt.subplots(figsize=(13, 8))
    for topic, sub in ratio_long.groupby("sync_topic"):
        sub = sub.set_index("period").reindex(PERIOD_ORDER).reset_index()
        ax.plot(sub["period"], sub["비율(%)"], marker="o", linewidth=2, label=topic)
    ax.set_title("발레 담론 토픽 비율 변화", fontsize=22, pad=14)
    ax.set_xlabel("시기")
    ax.set_ylabel("비율(%)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    plt.tight_layout()
    ratio_png = OUT_DIR / "topic_ratio_trend.png"
    plt.savefig(ratio_png, dpi=220)
    plt.close(fig)

    # export tables
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        count_tbl.to_excel(writer, sheet_name="topic_count_by_period", index=False)
        ratio_tbl.to_excel(writer, sheet_name="topic_ratio_by_period", index=False)
        count_long.to_excel(writer, sheet_name="topic_count_long", index=False)
        ratio_long.to_excel(writer, sheet_name="topic_ratio_long", index=False)

    print(f"완료: {count_png}")
    print(f"완료: {ratio_png}")
    print(f"완료: {OUT_XLSX}")


if __name__ == "__main__":
    main()
