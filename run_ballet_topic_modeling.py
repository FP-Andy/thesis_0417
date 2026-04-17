from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("/Users/andy/thesis")
INPUT_PATH = BASE_DIR / "ballet_articles_preprocessed_tfidf_filtered.xlsx"
OUTPUT_EXCEL_PATH = BASE_DIR / "ballet_lda_results.xlsx"
OUTPUT_MD_PATH = BASE_DIR / "ballet_lda_analysis_report.md"
VIS_DIR = BASE_DIR / "lda_vis"
VIS_DIR.mkdir(exist_ok=True)

# local target install path
LOCAL_PKG_PATH = Path("/tmp/thesis_pkgs")
if LOCAL_PKG_PATH.exists():
    sys.path.insert(0, str(LOCAL_PKG_PATH))

from gensim import corpora
from gensim.models import CoherenceModel, LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


PERIOD_ORDER = ["1기(~1995)", "2기(1996~2005)", "3기(2006~2015)", "4기(2016~2025)"]
K_RANGE = [4, 5, 6, 7, 8]
RANDOM_STATE = 4190
PASSES = 12
ITERATIONS = 120


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


def split_train_valid(doc_tokens: list[list[str]], valid_ratio: float = 0.2) -> tuple[list[list[str]], list[list[str]]]:
    idx = np.arange(len(doc_tokens))
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(idx)
    split = max(1, int(len(idx) * (1 - valid_ratio)))
    train_idx = idx[:split]
    valid_idx = idx[split:]
    train_docs = [doc_tokens[i] for i in train_idx]
    valid_docs = [doc_tokens[i] for i in valid_idx]
    if len(valid_docs) == 0:
        valid_docs = train_docs[: max(1, int(len(train_docs) * 0.1))]
    return train_docs, valid_docs


def get_theta_matrix(model: LdaModel, corpus: list[list[tuple[int, int]]], n_topics: int) -> np.ndarray:
    theta = np.zeros((len(corpus), n_topics), dtype=float)
    for i, bow in enumerate(corpus):
        for topic_id, prob in model.get_document_topics(bow, minimum_probability=0.0):
            theta[i, topic_id] = prob
    return theta


def sync_topic_name(top_words: list[str]) -> tuple[str, float]:
    best_topic = "기타"
    best_score = 0.0
    for name, lexicon in SYNC_TOPICS.items():
        score = sum(1.0 for w in top_words if w in lexicon)
        if score > best_score:
            best_score = score
            best_topic = name
    return best_topic, best_score


def format_topic_title(sync_name: str, top_words: list[str]) -> str:
    return f"{sync_name} ({'·'.join(top_words[:3])})"


def to_markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {INPUT_PATH}")

    raw_df = pd.read_excel(INPUT_PATH, sheet_name="preprocessed_articles")
    raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
    raw_df["keywords_filtered"] = raw_df["keywords_filtered"].fillna("").astype(str)
    raw_df["tokens"] = raw_df["keywords_filtered"].map(parse_keywords)
    raw_df = raw_df[raw_df["tokens"].map(len) >= 3].copy()

    selection_rows: list[dict] = []
    selected_rows: list[dict] = []
    topics_rows: list[dict] = []
    doc_topic_rows: list[dict] = []
    period_share_rows: list[dict] = []

    for period in PERIOD_ORDER:
        period_df = raw_df[raw_df["period"] == period].copy()
        docs = period_df["tokens"].tolist()
        if len(docs) < 50:
            continue

        train_docs, valid_docs = split_train_valid(docs, valid_ratio=0.2)
        dictionary = corpora.Dictionary(train_docs)
        dictionary.filter_extremes(no_below=5, no_above=0.5)
        train_corpus = [dictionary.doc2bow(d) for d in train_docs]
        valid_corpus = [dictionary.doc2bow(d) for d in valid_docs]

        k_scores = []
        for k in K_RANGE:
            start = time.time()
            model = LdaModel(
                corpus=train_corpus,
                id2word=dictionary,
                num_topics=k,
                random_state=RANDOM_STATE,
                passes=PASSES,
                iterations=ITERATIONS,
                alpha="auto",
                eta="auto",
            )
            elapsed = time.time() - start

            coherence = CoherenceModel(
                model=model,
                texts=train_docs,
                dictionary=dictionary,
                coherence="c_v",
                topn=10,
            ).get_coherence()

            train_log_perp = model.log_perplexity(train_corpus)
            valid_log_perp = model.log_perplexity(valid_corpus)
            train_perp = float(2 ** (-train_log_perp))
            valid_perp = float(2 ** (-valid_log_perp))

            row = {
                "period": period,
                "k": k,
                "coherence_c_v": round(float(coherence), 6),
                "train_log_perplexity": round(float(train_log_perp), 6),
                "valid_log_perplexity": round(float(valid_log_perp), 6),
                "train_perplexity": round(train_perp, 6),
                "valid_perplexity": round(valid_perp, 6),
                "fit_time_sec": round(float(elapsed), 3),
                "n_train_docs": len(train_docs),
                "n_valid_docs": len(valid_docs),
                "dictionary_size": len(dictionary),
            }
            selection_rows.append(row)
            k_scores.append(row)

        score_df = pd.DataFrame(k_scores).sort_values(
            by=["coherence_c_v", "valid_perplexity"], ascending=[False, True]
        )
        best = score_df.iloc[0].to_dict()
        selected_k = int(best["k"])
        selected_rows.append(
            {
                "period": period,
                "selected_k": selected_k,
                "selection_reason": "coherence 최대, valid perplexity 보조",
                "coherence_c_v": best["coherence_c_v"],
                "valid_perplexity": best["valid_perplexity"],
                "n_docs_used": len(docs),
            }
        )

        # final model on all docs in period
        final_dictionary = corpora.Dictionary(docs)
        final_dictionary.filter_extremes(no_below=5, no_above=0.5)
        final_corpus = [final_dictionary.doc2bow(d) for d in docs]
        final_model = LdaModel(
            corpus=final_corpus,
            id2word=final_dictionary,
            num_topics=selected_k,
            random_state=RANDOM_STATE,
            passes=PASSES,
            iterations=ITERATIONS,
            alpha="auto",
            eta="auto",
        )

        # pyLDAvis export
        vis_path = VIS_DIR / f"{period}_k{selected_k}.html"
        vis = gensimvis.prepare(
            final_model,
            final_corpus,
            final_dictionary,
            sort_topics=False,
            n_jobs=1,
        )
        pyLDAvis.save_html(vis, str(vis_path))

        theta = get_theta_matrix(final_model, final_corpus, selected_k)
        dominant_topic = theta.argmax(axis=1)
        topic_share = theta.mean(axis=0)

        period_df = period_df.reset_index(drop=True)
        for i in range(len(period_df)):
            doc_topic_rows.append(
                {
                    "period": period,
                    "doc_index_in_period": i,
                    "date": period_df.loc[i, "date"],
                    "title": period_df.loc[i, "title"],
                    "dominant_topic": int(dominant_topic[i]),
                    "dominant_topic_prob": round(float(theta[i, dominant_topic[i]]), 6),
                }
            )

        for topic_id in range(selected_k):
            top_terms = [w for w, _ in final_model.show_topic(topic_id, topn=10)]
            sync_name, sync_score = sync_topic_name(top_terms)
            topic_title = format_topic_title(sync_name, top_terms)

            # representative documents
            doc_idx = np.argsort(-theta[:, topic_id])[:3]
            rep_titles = [str(period_df.loc[j, "title"]) for j in doc_idx]
            rep_dates = [str(pd.to_datetime(period_df.loc[j, "date"]).date()) for j in doc_idx]

            topics_rows.append(
                {
                    "period": period,
                    "topic_id": topic_id,
                    "topic_title": topic_title,
                    "sync_topic": sync_name,
                    "sync_score": sync_score,
                    "topic_share_percent": round(float(topic_share[topic_id] * 100), 3),
                    "top_keywords": ", ".join(top_terms),
                    "representative_titles": " | ".join(rep_titles),
                    "representative_dates": " | ".join(rep_dates),
                }
            )

            period_share_rows.append(
                {
                    "period": period,
                    "sync_topic": sync_name,
                    "topic_share_percent": round(float(topic_share[topic_id] * 100), 3),
                }
            )

    # post aggregation
    selection_df = pd.DataFrame(selection_rows)
    selected_df = pd.DataFrame(selected_rows)
    topics_df = pd.DataFrame(topics_rows)
    doc_topic_df = pd.DataFrame(doc_topic_rows)
    period_share_df = pd.DataFrame(period_share_rows)

    sync_summary_df = (
        period_share_df.groupby(["period", "sync_topic"], as_index=False)["topic_share_percent"]
        .sum()
        .sort_values(["period", "topic_share_percent"], ascending=[True, False])
    )
    pattern_pivot_df = (
        sync_summary_df.pivot(index="sync_topic", columns="period", values="topic_share_percent")
        .fillna(0.0)
        .reset_index()
    )

    with pd.ExcelWriter(OUTPUT_EXCEL_PATH, engine="openpyxl") as writer:
        selection_df.to_excel(writer, sheet_name="model_selection_metrics", index=False)
        selected_df.to_excel(writer, sheet_name="selected_k", index=False)
        topics_df.to_excel(writer, sheet_name="topics_by_period", index=False)
        sync_summary_df.to_excel(writer, sheet_name="sync_topic_share", index=False)
        pattern_pivot_df.to_excel(writer, sheet_name="sync_topic_pattern", index=False)
        doc_topic_df.to_excel(writer, sheet_name="doc_dominant_topic", index=False)

    # markdown report
    md_lines = []
    md_lines.append("# 발레 미디어 담론 LDA 분석 결과")
    md_lines.append("")
    md_lines.append("## 1) 분석 개요")
    md_lines.append("- 데이터: `ballet_articles_preprocessed_tfidf_filtered.xlsx`")
    md_lines.append("- 입력 문서: TF-IDF 상투어 제거 후 토큰(`keywords_filtered`) 사용")
    md_lines.append("- 상투어 제거: `발레` 고정 제거 + TF-IDF 기반 자동 제거")
    md_lines.append("- 시기 구분: 1기(~1995), 2기(1996~2005), 3기(2006~2015), 4기(2016~2025)")
    md_lines.append("- 토픽 수 탐색: 기별 `k=4~8`")
    md_lines.append("- 정량 기준: coherence(c_v), holdout(valid) perplexity")
    md_lines.append("- 정성 기준: 연구자 해석 가능성(대표 키워드/대표 기사 확인)")
    md_lines.append("")
    md_lines.append("MathWorks 가이드(토픽 수 선택)에서 제시한 정량 점검 취지를 반영해,")
    md_lines.append("기별로 holdout perplexity와 coherence를 함께 비교한 뒤 최종 k를 선정했다.")
    md_lines.append("")

    md_lines.append("## 2) 기별 최종 k")
    md_lines.append(to_markdown_table(selected_df))
    md_lines.append("")

    md_lines.append("## 3) k 탐색 지표(요약)")
    md_lines.append(to_markdown_table(selection_df.sort_values(["period", "k"])))
    md_lines.append("")

    md_lines.append("## 4) 기별 토픽 결과")
    for period in PERIOD_ORDER:
        sub = topics_df[topics_df["period"] == period].copy()
        if len(sub) == 0:
            continue
        md_lines.append(f"### {period}")
        keep = sub[
            ["topic_id", "topic_title", "sync_topic", "topic_share_percent", "top_keywords", "representative_titles"]
        ].copy()
        md_lines.append(to_markdown_table(keep))
        md_lines.append("")

    md_lines.append("## 5) 상위 토픽 동기화 패턴(기별 합산 점유율, %)")
    md_lines.append(to_markdown_table(pattern_pivot_df))
    md_lines.append("")

    md_lines.append("## 6) 산출물 경로")
    md_lines.append(f"- 엑셀 결과: `{OUTPUT_EXCEL_PATH}`")
    md_lines.append(f"- 시각화 HTML 폴더: `{VIS_DIR}`")
    md_lines.append("- 시트 구성: `model_selection_metrics`, `selected_k`, `topics_by_period`, `sync_topic_share`, `sync_topic_pattern`, `doc_dominant_topic`")
    md_lines.append("")

    OUTPUT_MD_PATH.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"완료: {OUTPUT_EXCEL_PATH}")
    print(f"완료: {OUTPUT_MD_PATH}")
    print(f"시각화 폴더: {VIS_DIR}")
    print("선정 k:")
    print(selected_df.to_string(index=False))


if __name__ == "__main__":
    main()
