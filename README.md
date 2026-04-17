# Ballet Media Discourse Topic Modeling (1980-2025)

This repository contains the end-to-end Python workflow and outputs for period-based ballet media discourse analysis.

## Main Outputs

- `ballet_lda_analysis_report.md`: analysis report draft
- `ballet_lda_results.xlsx`: model selection, topic tables, synchronized patterns
- `ballet_topic_table_reference_style.xlsx`: reference-style topic table
- `topic_network_graphs/`: period-wise topic-keyword network figures
- `topic_trend_graphs/`: period-wise trend graphs (count/ratio)

## Key Scripts

- `build_ballet_master_dataset.py`
- `preprocess_ballet_for_topic_modeling.py`
- `apply_tfidf_banal_filter.py`
- `run_ballet_topic_modeling.py`
- `relabel_topics_unique.py`
- `build_reference_style_topic_table.py`
- `build_topic_network_graphs.py`
- `build_period_trend_graphs.py`

## Data Notes

- Source files include BigKinds exports and Chosun crawling data.
- Analysis periods: `~1995`, `1996~2005`, `2006~2015`, `2016~2025`.
