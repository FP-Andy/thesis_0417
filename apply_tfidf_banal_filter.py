from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("/Users/andy/thesis")
INPUT_PATH = BASE_DIR / "ballet_articles_preprocessed.xlsx"
OUTPUT_PATH = BASE_DIR / "ballet_articles_preprocessed_tfidf_filtered.xlsx"
OUTPUT_STOPWORDS_PATH = BASE_DIR / "ballet_tfidf_stopwords.csv"

# local target install path
LOCAL_PKG_PATH = Path("/tmp/thesis_pkgs")
if LOCAL_PKG_PATH.exists():
    sys.path.insert(0, str(LOCAL_PKG_PATH))

from sklearn.feature_extraction.text import TfidfVectorizer


def tokenize(line: str) -> list[str]:
    if not isinstance(line, str):
        return []
    return [w for w in line.split() if w]


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {INPUT_PATH}")

    df = pd.read_excel(INPUT_PATH, sheet_name="preprocessed_articles")
    df["keywords"] = df["keywords"].fillna("").astype(str)
    token_lists = df["keywords"].map(tokenize)
    texts = token_lists.map(lambda xs: " ".join(xs))

    # TF-IDF 기준:
    # - 문서에 너무 널리 등장(df_ratio 높음)
    # - 역문서빈도(IDF) 낮음
    # 레퍼런스 논문의 '상투어 제거' 의도를 반영
    df_ratio_threshold = 0.15
    idf_threshold = 2.8

    # 강제 제외어: 연구키워드라 정보성이 낮은 단어
    force_remove = {"발레"}

    # 보호어: 상투어처럼 보일 수 있어도 연구 해석상 핵심인 단어
    protected_terms = {
        "발레단",
        "유아발레",
        "성인발레",
        "시니어발레",
        "창작발레",
        "클래식발레",
        "모던발레",
        "컨템포러리발레",
        "콩쿠르",
    }

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        use_idf=True,
        smooth_idf=True,
    )
    X = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())
    idf = vectorizer.idf_
    doc_freq = (X > 0).sum(axis=0).A1
    doc_freq_ratio = doc_freq / X.shape[0]

    auto_stop_mask = (doc_freq_ratio >= df_ratio_threshold) & (idf <= idf_threshold)
    auto_stopwords = set(terms[auto_stop_mask].tolist())

    final_stopwords = (auto_stopwords - protected_terms) | force_remove

    stat_df = pd.DataFrame(
        {
            "term": terms,
            "doc_freq": doc_freq,
            "doc_freq_ratio": doc_freq_ratio,
            "idf": idf,
        }
    )
    stat_df["is_auto_stopword"] = stat_df["term"].isin(auto_stopwords)
    stat_df["is_protected"] = stat_df["term"].isin(protected_terms)
    stat_df["is_forced_remove"] = stat_df["term"].isin(force_remove)
    stat_df["is_final_stopword"] = stat_df["term"].isin(final_stopwords)
    stat_df = stat_df.sort_values(["is_final_stopword", "doc_freq_ratio", "idf"], ascending=[False, False, True])

    stopword_set = set(final_stopwords)
    df["keywords_filtered"] = token_lists.map(lambda xs: [w for w in xs if w not in stopword_set]).map(" ".join)
    df["noun_count_before"] = token_lists.map(len)
    df["noun_count_after"] = df["keywords_filtered"].map(lambda x: len(tokenize(x)))

    removed_total = int((df["noun_count_before"] - df["noun_count_after"]).sum())

    # 결과 저장
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="preprocessed_articles", index=False)
        stat_df.to_excel(writer, sheet_name="tfidf_term_stats", index=False)
        pd.DataFrame({"stopword": sorted(stopword_set)}).to_excel(
            writer, sheet_name="tfidf_stopwords", index=False
        )

    pd.DataFrame({"stopword": sorted(stopword_set)}).to_csv(
        OUTPUT_STOPWORDS_PATH, index=False, encoding="utf-8-sig"
    )

    print(f"입력 문서 수: {len(df):,}")
    print(f"자동 상투어 후보 수: {len(auto_stopwords):,}")
    print(f"최종 제거어 수(강제 포함): {len(stopword_set):,}")
    print(f"제거된 토큰 총합: {removed_total:,}")
    print(f"최종 결과 파일: {OUTPUT_PATH}")
    print(f"제거어 목록 파일: {OUTPUT_STOPWORDS_PATH}")
    print("최종 제거어:")
    print(", ".join(sorted(stopword_set)))


if __name__ == "__main__":
    main()
