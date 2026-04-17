from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


BASE_DIR = Path("/Users/andy/thesis")
OUTPUT_PATH = BASE_DIR / "ballet_articles_master_dedup.xlsx"


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def parse_bigkinds_date(series: pd.Series) -> pd.Series:
    yyyy_mm_dd = (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.extract(r"(\d{8})", expand=False)
    )
    return pd.to_datetime(yyyy_mm_dd, format="%Y%m%d", errors="coerce")


def map_period(year: int) -> str:
    if year <= 1995:
        return "1기(~1995)"
    if 1996 <= year <= 2005:
        return "2기(1996~2005)"
    if 2006 <= year <= 2015:
        return "3기(2006~2015)"
    return "4기(2016~2025)"


def load_bigkinds(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=0)
    out = pd.DataFrame(
        {
            "date": parse_bigkinds_date(df["일자"]),
            "title": df["제목"].astype(str),
            "content": df["본문"].astype(str),
            "source": "BIG_KINDS",
            "source_file": file_path.name,
        }
    )
    return out


def load_chosun(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=0)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["날짜"], errors="coerce"),
            "title": df["제목"].astype(str),
            "content": df["본문"].astype(str),
            "source": "CHOSUN_CRAWL",
            "source_file": file_path.name,
        }
    )
    return out


def main() -> None:
    all_frames: list[pd.DataFrame] = []

    bigkinds_files = sorted(
        p for p in BASE_DIR.glob("NewsResult_*.xlsx") if not p.name.startswith("~$")
    )
    for fp in bigkinds_files:
        all_frames.append(load_bigkinds(fp))

    chosun_path = BASE_DIR / "chosun_ballet_articles_1980_1990.xlsx"
    if chosun_path.exists():
        all_frames.append(load_chosun(chosun_path))

    if not all_frames:
        raise FileNotFoundError("입력 엑셀 파일을 찾지 못했습니다.")

    df = pd.concat(all_frames, ignore_index=True)

    df["date_norm"] = df["date"].dt.strftime("%Y-%m-%d").fillna("")
    df["title_norm"] = normalize_text(df["title"])
    df["content_norm"] = normalize_text(df["content"])

    before_count = len(df)
    dedup_df = df.drop_duplicates(
        subset=["date_norm", "title_norm", "content_norm"], keep="first"
    ).copy()
    after_count = len(dedup_df)

    dedup_df["year"] = dedup_df["date"].dt.year.astype("Int64")
    dedup_df["period"] = dedup_df["year"].apply(
        lambda y: map_period(int(y)) if pd.notna(y) else "UNKNOWN"
    )

    final_df = dedup_df[
        ["date", "year", "period", "title", "content", "source", "source_file"]
    ].sort_values(["date", "title"], na_position="last")

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="articles", index=False)

    removed_count = before_count - after_count
    print(f"입력 기사 수: {before_count:,}")
    print(f"중복 제거 후 기사 수: {after_count:,}")
    print(f"제거된 중복 건수: {removed_count:,}")
    print(f"저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
