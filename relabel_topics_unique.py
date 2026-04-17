from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path("/Users/andy/thesis")
RESULT_XLSX = BASE_DIR / "ballet_lda_results.xlsx"
REFERENCE_XLSX = BASE_DIR / "ballet_topic_table_reference_style.xlsx"


# 기수 내 토픽명 중복 제거를 위한 수동 재명명(검토 기반)
REVISED_TOPIC_NAMES = {
    "1기(~1995)": {
        0: "국제 스타 내한·러시아 교류",
        1: "발레단 및 기관 기반 형성",
        2: "국제 콩쿠르·페스티벌 확산",
        3: "가족·연말 레퍼토리 공연",
        4: "수중발레·체육대회 이슈",
        5: "명작·갈라 중심 무대",
        6: "창작발레·안무 교류",
    },
    "2기(1996~2005)": {
        0: "콩쿠르·인재양성 담론",
        1: "클래식 레퍼토리 정기공연",
        2: "가족·연말 발레향유",
        3: "발레 교육·원로 담론",
        4: "글로벌 스타 갈라 교류",
        5: "창작발레·현대공연 확장",
        6: "지역 발레단 레퍼토리",
    },
    "3기(2006~2015)": {
        0: "방송·셀럽 중심 대중화",
        1: "가족 레퍼토리 공연",
        2: "콩쿠르·경연 성과 담론",
        3: "스타 발레단 브랜드화",
        4: "국제·창작 공연장 확장",
    },
    "4기(2016~2025)": {
        0: "드라마·대중미디어 확산",
        1: "지역기관·생활프로그램",
        2: "공연·축제 이벤트형 담론",
        3: "국제 콩쿠르 성과",
        4: "글로벌 스타·대형공연",
        5: "국제교류·지역확산",
    },
}


def main() -> None:
    if not RESULT_XLSX.exists():
        raise FileNotFoundError(RESULT_XLSX)

    topics_df = pd.read_excel(RESULT_XLSX, sheet_name="topics_by_period")

    topics_df["토픽명_수정"] = topics_df.apply(
        lambda r: REVISED_TOPIC_NAMES.get(r["period"], {}).get(int(r["topic_id"]), r["topic_title"]),
        axis=1,
    )

    # 검증: 기수 내 중복 토픽명 여부
    dup_check = (
        topics_df.groupby(["period", "토픽명_수정"]).size().reset_index(name="n").query("n > 1")
    )
    if len(dup_check) > 0:
        raise ValueError(f"수정 토픽명 중복 발생: {dup_check.to_dict(orient='records')}")

    # 레퍼런스 스타일 표 재생성 (점유율 기준)
    ref_df = (
        topics_df[["period", "토픽명_수정", "topic_share_percent", "top_keywords"]]
        .rename(
            columns={
                "period": "시기",
                "토픽명_수정": "토픽",
                "topic_share_percent": "점유율(%)",
                "top_keywords": "키워드(확률분포)",
            }
        )
        .sort_values(["시기", "점유율(%)"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # 결과 파일 업데이트
    with pd.ExcelWriter(RESULT_XLSX, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        topics_df.to_excel(writer, sheet_name="topics_by_period", index=False)
        ref_df.to_excel(writer, sheet_name="표1_기별토픽모델링결과", index=False)

    with pd.ExcelWriter(REFERENCE_XLSX, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        ref_df.to_excel(writer, sheet_name="표1_기별토픽모델링결과", index=False)

    print("완료:")
    print(f"- {RESULT_XLSX}")
    print(f"- {REFERENCE_XLSX}")
    print(f"수정 토픽 수: {len(ref_df)}")


if __name__ == "__main__":
    main()
