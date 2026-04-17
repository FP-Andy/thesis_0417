from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


BASE_DIR = Path("/Users/andy/thesis")
INPUT_PATH = BASE_DIR / "ballet_articles_master_dedup.xlsx"
OUTPUT_ARTICLES_PATH = BASE_DIR / "ballet_articles_preprocessed.xlsx"
OUTPUT_KEYWORDS_PATH = BASE_DIR / "ballet_keyword_frequency.csv"


# local target install path (for this environment)
LOCAL_PKG_PATH = Path("/tmp/thesis_pkgs")
if LOCAL_PKG_PATH.exists():
    sys.path.insert(0, str(LOCAL_PKG_PATH))

try:
    from kiwipiepy import Kiwi
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "kiwipiepy가 필요합니다. "
        "설치 예시: python3 -m pip install --target /tmp/thesis_pkgs kiwipiepy"
    ) from exc


DESIGNATED_TERMS = [
    "유아발레",
    "성인발레",
    "시니어발레",
    "창작발레",
    "클래식발레",
    "모던발레",
    "컨템포러리발레",
    "콩쿠르",
    "발레단",
]


SYNONYM_REPLACEMENTS = [
    (r"취미\s*발레", "성인발레"),
    (r"고전\s*발레", "클래식발레"),
    (r"콩쿨", "콩쿠르"),
]


REGEX_NORMALIZATION_RULES = [
    # user-requested pattern-based normalization
    (r"유아(?:\s|의|대상|전용|를\s*위한|을\s*위한){0,4}발레", "유아발레"),
    (r"성인(?:\s|의|대상|전용|를\s*위한|을\s*위한){0,4}발레", "성인발레"),
    (r"시니어(?:\s|의|대상|전용|를\s*위한|을\s*위한){0,4}발레", "시니어발레"),
    # spacing variants
    (r"창작\s*발레", "창작발레"),
    (r"클래식\s*발레", "클래식발레"),
    (r"모던\s*발레", "모던발레"),
    (r"컨템포러리\s*발레", "컨템포러리발레"),
]


BALLET_COMPANY_RULES = [
    # explicit mapping list from project requirement
    (r"국립\s*발레단", "발레단"),
    (r"유니버설\s*발레단", "발레단"),
    (r"유니버셜\s*발레단", "발레단"),
    (r"서울시\s*발레단", "발레단"),
    (r"인천시티\s*발레단", "발레단"),
    (r"서울시어터", "발레단"),
    (r"와이즈\s*발레단", "발레단"),
    (r"광주시립\s*발레단", "발레단"),
    # generic company normalization
    (r"[가-힣A-Za-z]{1,12}(?:시립|도립|구립)?\s*발레단", "발레단"),
]


STOPWORDS = {
    "기자",
    "이번",
    "관련",
    "통해",
    "위해",
    "지난",
    "오전",
    "오후",
    "뉴스",
    "사진",
    "영상",
}


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def apply_replacements(text: str, rules: list[tuple[str, str]]) -> str:
    out = text
    for pattern, replacement in rules:
        out = re.sub(pattern, replacement, out)
    return out


def normalize_document(text: str) -> str:
    out = str(text)
    out = apply_replacements(out, SYNONYM_REPLACEMENTS)
    out = apply_replacements(out, REGEX_NORMALIZATION_RULES)
    out = apply_replacements(out, BALLET_COMPANY_RULES)
    return normalize_whitespace(out)


def is_valid_noun(token: str, tag: str) -> bool:
    if tag not in {"NNG", "NNP"}:
        return False
    if len(token) <= 1:
        return False
    if token in STOPWORDS:
        return False
    return True


def extract_nouns(text: str, kiwi: Kiwi) -> list[str]:
    tokens = kiwi.tokenize(text)
    nouns = [tok.form for tok in tokens if is_valid_noun(tok.form, tok.tag)]
    return nouns


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {INPUT_PATH}")

    df = pd.read_excel(INPUT_PATH, sheet_name="articles")
    df["title"] = df["title"].fillna("").astype(str)
    df["content"] = df["content"].fillna("").astype(str)
    df["doc_text"] = (df["title"] + " " + df["content"]).str.strip()

    print(f"입력 기사 수: {len(df):,}")

    kiwi = Kiwi()
    for term in DESIGNATED_TERMS:
        kiwi.add_user_word(term, tag="NNP")

    df["normalized_text"] = df["doc_text"].apply(normalize_document)
    df["nouns"] = df["normalized_text"].apply(lambda x: extract_nouns(x, kiwi))
    df["noun_count"] = df["nouns"].apply(len)
    df["keywords"] = df["nouns"].apply(lambda xs: " ".join(xs))

    keyword_counter: Counter[str] = Counter()
    for words in df["nouns"]:
        keyword_counter.update(words)

    keyword_df = (
        pd.DataFrame(keyword_counter.items(), columns=["keyword", "freq"])
        .sort_values("freq", ascending=False)
        .reset_index(drop=True)
    )

    out_df = df[
        [
            "date",
            "year",
            "period",
            "source",
            "source_file",
            "title",
            "content",
            "normalized_text",
            "keywords",
            "noun_count",
        ]
    ]

    with pd.ExcelWriter(OUTPUT_ARTICLES_PATH, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="preprocessed_articles", index=False)
        keyword_df.to_excel(writer, sheet_name="keyword_frequency", index=False)

    keyword_df.to_csv(OUTPUT_KEYWORDS_PATH, index=False, encoding="utf-8-sig")

    print(f"전처리 결과 저장: {OUTPUT_ARTICLES_PATH}")
    print(f"키워드 빈도 저장: {OUTPUT_KEYWORDS_PATH}")
    print("상위 20개 키워드:")
    print(keyword_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
