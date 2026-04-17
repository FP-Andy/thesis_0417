from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path("/Users/andy/thesis")
INPUT_PATH = BASE_DIR / "ballet_articles_preprocessed_tfidf_filtered.xlsx"
RESULT_PATH = BASE_DIR / "ballet_lda_results.xlsx"
OUTPUT_PATH = BASE_DIR / "ballet_topic_table_reference_style.xlsx"

LOCAL_PKG_PATH = Path("/tmp/thesis_pkgs")
if LOCAL_PKG_PATH.exists():
    sys.path.insert(0, str(LOCAL_PKG_PATH))

from gensim import corpora
from gensim.models import LdaModel


RANDOM_STATE = 4190
PASSES = 12
ITERATIONS = 120
PERIOD_ORDER = ["1기(~1995)", "2기(1996~2005)", "3기(2006~2015)", "4기(2016~2025)"]

SYNC_TOPICS = {
    "공연 및 축제": {"공연", "축제", "페스티벌", "무대", "관객", "극장", "회관", "오페라", "갈라", "클래식발레"},
    "발레단 및 기관": {"발레단", "국립", "시립", "재단", "협회", "센터", "예술의전당", "단장"},
    "발레 교육": {"교육", "학교", "학생", "교수", "아카데미", "수업", "청소년", "강좌"},
    "대중화/취미발레": {"성인발레", "취미", "직장인", "클래스", "생활", "센터", "주민"},
    "세대별 발레 참여": {"유아발레", "시니어발레", "어린이", "가족", "노인", "키즈"},
    "국제 교류": {"국제", "세계", "해외", "교류", "러시아", "프랑스", "유럽", "초청"},
    "창작 및 현대화": {"창작발레", "컨템포러리발레", "모던발레", "안무", "현대", "창작", "실험"},
    "콩쿠르/경연": {"콩쿠르", "대회", "경연", "입상", "수상", "참가"},
    "산업 및 콘텐츠화": {"콘텐츠", "산업", "미디어", "관광", "브랜드", "마케팅", "플랫폼", "흥행"},
}


def parse_keywords(line: str) -> list[str]:
    if not isinstance(line, str):
        return []
    return [w for w in line.split() if w]


def sync_topic_name(top_words: list[str]) -> str:
    best_topic = "기타"
    best_score = -1.0
    for name, lexicon in SYNC_TOPICS.items():
        score = sum(1.0 for w in top_words if w in lexicon)
        if score > best_score:
            best_score = score
            best_topic = name
    return best_topic


def get_theta_matrix(model: LdaModel, corpus: list[list[tuple[int, int]]], n_topics: int) -> np.ndarray:
    theta = np.zeros((len(corpus), n_topics), dtype=float)
    for i, bow in enumerate(corpus):
        for topic_id, prob in model.get_document_topics(bow, minimum_probability=0.0):
            theta[i, topic_id] = prob
    return theta


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)
    if not RESULT_PATH.exists():
        raise FileNotFoundError(RESULT_PATH)

    df = pd.read_excel(INPUT_PATH, sheet_name="preprocessed_articles")
    df["tokens"] = df["keywords_filtered"].fillna("").map(parse_keywords)
    df = df[df["tokens"].map(len) >= 3].copy()

    selected = pd.read_excel(RESULT_PATH, sheet_name="selected_k")
    selected_k_map = dict(zip(selected["period"], selected["selected_k"]))

    rows = []
    for period in PERIOD_ORDER:
        period_df = df[df["period"] == period].copy()
        if len(period_df) == 0:
            continue
        docs = period_df["tokens"].tolist()

        dictionary = corpora.Dictionary(docs)
        dictionary.filter_extremes(no_below=5, no_above=0.5)
        corpus = [dictionary.doc2bow(d) for d in docs]

        k = int(selected_k_map[period])
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
        theta = get_theta_matrix(model, corpus, k)
        topic_share = theta.mean(axis=0) * 100

        # 점유율 기준 내림차순 정렬
        topic_order = np.argsort(-topic_share)
        for topic_id in topic_order:
            topic_terms = model.show_topic(int(topic_id), topn=10)
            top_words = [w for w, _ in topic_terms]
            sync_name = sync_topic_name(top_words)
            keyword_prob_text = ", ".join([f"{w}({p:.3f})" for w, p in topic_terms])

            rows.append(
                {
                    "시기": period,
                    "토픽": sync_name,
                    "점유율(%)": round(float(topic_share[topic_id]), 2),
                    "키워드(확률분포)": keyword_prob_text,
                }
            )

    out_df = pd.DataFrame(rows)

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="표1_기별토픽모델링결과", index=False)

    print(f"완료: {OUTPUT_PATH}")
    print(f"행 수: {len(out_df)}")


if __name__ == "__main__":
    main()
