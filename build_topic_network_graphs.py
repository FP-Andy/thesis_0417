from __future__ import annotations

import math
import os
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

BASE_DIR = Path("/Users/andy/thesis")
INPUT_PATH = BASE_DIR / "ballet_articles_preprocessed_tfidf_filtered.xlsx"
RESULT_PATH = BASE_DIR / "ballet_lda_results.xlsx"
OUTPUT_DIR = BASE_DIR / "topic_network_graphs"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DATA_PATH = BASE_DIR / "ballet_topic_network_edges.xlsx"

LOCAL_PKG_PATH = Path("/tmp/thesis_pkgs")
if LOCAL_PKG_PATH.exists():
    sys.path.insert(0, str(LOCAL_PKG_PATH))

from gensim import corpora
from gensim.models import LdaModel

# matplotlib cache dir workaround
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx


RANDOM_STATE = 4190
PASSES = 12
ITERATIONS = 120
PERIOD_ORDER = ["1기(~1995)", "2기(1996~2005)", "3기(2006~2015)", "4기(2016~2025)"]


def parse_tokens(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return [w.strip() for w in text.split() if w.strip()]


FONT_PROP = None
FONT_NAME = None


def set_korean_font() -> None:
    global FONT_PROP, FONT_NAME
    # Explicit font file path first to avoid missing-glyph issues
    candidates = [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/NanumGothic.ttf",
    ]
    selected_name = None
    for fpath in candidates:
        if Path(fpath).exists():
            try:
                fm.fontManager.addfont(fpath)
                FONT_PROP = fm.FontProperties(fname=fpath)
                selected_name = FONT_PROP.get_name()
                break
            except Exception:
                continue
    if selected_name:
        FONT_NAME = selected_name
        plt.rcParams["font.family"] = [selected_name, "sans-serif"]
    else:
        FONT_NAME = None
        plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def topic_positions(n_topics: int, radius: float = 3.0) -> dict[int, tuple[float, float]]:
    pos = {}
    for i in range(n_topics):
        ang = 2 * math.pi * (i / n_topics)
        pos[i] = (radius * math.cos(ang), radius * math.sin(ang))
    return pos


def draw_period_network(
    period: str,
    topic_edges: list[tuple[int, str, float]],
    topic_labels: dict[int, str],
    out_png: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    G = nx.Graph()
    topic_ids = sorted(set(t for t, _, _ in topic_edges))

    # add nodes
    for t in topic_ids:
        G.add_node(f"T{t}", node_type="topic", label=topic_labels[t])
    keywords = sorted(set(k for _, k, _ in topic_edges))
    for k in keywords:
        G.add_node(f"K:{k}", node_type="keyword", label=k)

    # add edges
    for t, k, p in topic_edges:
        G.add_edge(f"T{t}", f"K:{k}", weight=float(p))

    # positions
    t_pos_raw = topic_positions(len(topic_ids), radius=3.0)
    t_pos = {f"T{t}": t_pos_raw[idx] for idx, t in enumerate(topic_ids)}

    k_pos = {}
    # 1) exclusive keywords: distribute evenly in circle around each topic
    exclusive_by_topic = defaultdict(list)
    # 2) shared keywords: place near topic-centroid with local spreading
    shared_group = defaultdict(list)
    for kw in keywords:
        neighbors = [n for n in G.neighbors(f"K:{kw}") if n.startswith("T")]
        if len(neighbors) == 1:
            tid = int(neighbors[0].replace("T", ""))
            exclusive_by_topic[tid].append(kw)
        else:
            # shared keyword: group by same connected topics, then spread to avoid overlap
            key = tuple(sorted(neighbors))
            shared_group[key].append(kw)

    # exclusive keywords evenly spread around topic center
    for tid in topic_ids:
        kw_list = sorted(exclusive_by_topic.get(tid, []))
        if not kw_list:
            continue
        tx, ty = t_pos[f"T{tid}"]
        n = len(kw_list)
        # slight rotation by topic id to avoid all clusters aligned similarly
        base = (tid * 0.55) % (2 * math.pi)
        radius = 1.55 if n <= 10 else 1.75
        for i, kw in enumerate(kw_list):
            ang = base + (2 * math.pi * i / n)
            k_pos[f"K:{kw}"] = (tx + radius * math.cos(ang), ty + radius * math.sin(ang))

    for neigh_tuple, kw_list in shared_group.items():
        xs = [t_pos[n][0] for n in neigh_tuple]
        ys = [t_pos[n][1] for n in neigh_tuple]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        n = len(kw_list)
        for i, kw in enumerate(sorted(kw_list)):
            ang = 2 * math.pi * (i / max(n, 1))
            rr = 0.22 + 0.08 * min(n, 8)
            k_pos[f"K:{kw}"] = (cx + rr * math.cos(ang), cy + rr * math.sin(ang))

    pos = {}
    pos.update(t_pos)
    pos.update(k_pos)

    # draw
    set_korean_font()
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_facecolor("#f2f2f2")
    fig.patch.set_facecolor("#f2f2f2")

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    if edge_weights:
        w_min, w_max = min(edge_weights), max(edge_weights)
    else:
        w_min, w_max = 0.0, 1.0

    def scale_width(w: float) -> float:
        if w_max == w_min:
            return 2.0
        return 0.8 + (w - w_min) / (w_max - w_min) * 6.0

    widths = [scale_width(w) for w in edge_weights]
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="#7f7f7f", alpha=0.75, ax=ax)

    topic_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "topic"]
    keyword_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "keyword"]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=topic_nodes,
        node_shape="s",
        node_size=420,
        node_color="#3b6fb6",
        edgecolors="white",
        linewidths=1.2,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=keyword_nodes,
        node_shape="o",
        node_size=68,
        node_color="#c95f4a",
        edgecolors="white",
        linewidths=0.6,
        ax=ax,
    )

    labels = {n: d["label"] for n, d in G.nodes(data=True)}
    # draw labels manually with offsets for readability
    for node in topic_nodes:
        x, y = pos[node]
        ax.text(
            x,
            y - 0.22,
            labels[node],
            fontsize=12,
            color="#2f5f9d",
            ha="center",
            va="center",
            fontproperties=FONT_PROP,
            bbox=dict(facecolor="#f2f2f2", edgecolor="none", pad=0.2, alpha=0.95),
        )
    for node in keyword_nodes:
        x, y = pos[node]
        ax.text(
            x + 0.07,
            y + 0.06,
            labels[node],
            fontsize=9,
            color="#c95f4a",
            ha="left",
            va="center",
            fontproperties=FONT_PROP,
            bbox=dict(facecolor="#f2f2f2", edgecolor="none", pad=0.1, alpha=0.9),
        )

    ax.set_title(f"{period} 토픽-키워드 네트워크", fontsize=18, color="#2f2f2f", pad=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)

    # return data tables
    edge_rows = []
    for u, v, d in G.edges(data=True):
        if u.startswith("T"):
            topic_node, keyword_node = u, v
        else:
            topic_node, keyword_node = v, u
        edge_rows.append(
            {
                "period": period,
                "topic_id": int(topic_node.replace("T", "")),
                "topic_name": labels[topic_node],
                "keyword": labels[keyword_node],
                "keyword_prob": d["weight"],
            }
        )
    edges_df = pd.DataFrame(edge_rows).sort_values(["topic_id", "keyword_prob"], ascending=[True, False])

    shared_kw = (
        edges_df.groupby("keyword")["topic_id"].nunique().reset_index(name="connected_topic_count")
    )
    shared_kw = shared_kw[shared_kw["connected_topic_count"] > 1].sort_values(
        "connected_topic_count", ascending=False
    )
    shared_kw.insert(0, "period", period)
    return edges_df, shared_kw


def main() -> None:
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)
    if not RESULT_PATH.exists():
        raise FileNotFoundError(RESULT_PATH)

    input_df = pd.read_excel(INPUT_PATH, sheet_name="preprocessed_articles")
    topics_df = pd.read_excel(RESULT_PATH, sheet_name="topics_by_period")
    selected_df = pd.read_excel(RESULT_PATH, sheet_name="selected_k")

    input_df["tokens"] = input_df["keywords_filtered"].fillna("").map(parse_tokens)
    input_df = input_df[input_df["tokens"].map(len) >= 3].copy()

    selected_k = dict(zip(selected_df["period"], selected_df["selected_k"]))

    all_edges = []
    all_shared = []

    for period in PERIOD_ORDER:
        period_docs = input_df[input_df["period"] == period]["tokens"].tolist()
        if not period_docs:
            continue

        dictionary = corpora.Dictionary(period_docs)
        dictionary.filter_extremes(no_below=5, no_above=0.5)
        corpus = [dictionary.doc2bow(d) for d in period_docs]
        k = int(selected_k[period])

        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=RANDOM_STATE,
            passes=PASSES,
            iterations=ITERATIONS,
            alpha="auto",
            eta="auto",
        )

        t_sub = topics_df[topics_df["period"] == period].copy()
        topic_name_map = {
            int(r["topic_id"]): str(r["토픽명_수정"]) if pd.notna(r.get("토픽명_수정")) else str(r["topic_title"])
            for _, r in t_sub.iterrows()
        }

        edges = []
        for tid in range(k):
            top_terms = model.show_topic(tid, topn=10)
            for word, prob in top_terms:
                edges.append((tid, word, float(prob)))

        out_png = OUTPUT_DIR / f"{period}_topic_network.png"
        edges_df, shared_df = draw_period_network(period, edges, topic_name_map, out_png)
        all_edges.append(edges_df)
        all_shared.append(shared_df)

    edges_all_df = pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame()
    shared_all_df = pd.concat(all_shared, ignore_index=True) if all_shared else pd.DataFrame()

    with pd.ExcelWriter(OUTPUT_DATA_PATH, engine="openpyxl") as writer:
        edges_all_df.to_excel(writer, sheet_name="topic_keyword_edges", index=False)
        shared_all_df.to_excel(writer, sheet_name="shared_keywords", index=False)

    print(f"완료: {OUTPUT_DIR}")
    print(f"완료: {OUTPUT_DATA_PATH}")
    if not shared_all_df.empty:
        print("중복(공유) 키워드 상위:")
        print(shared_all_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
