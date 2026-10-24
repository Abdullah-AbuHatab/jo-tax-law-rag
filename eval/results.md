# Evaluation results

- mode: `real`
- date: `2026-10-24 20:46 UTC`
- top_k: `5`
- min_similarity: `0.6`
- embedder: `gemini-embedding-001`
- llm: `nvidia/nemotron-3-ultra-550b-a55b:free, cohere/north-mini-code:free, google/gemma-4-31b-it:free`
- golden: `data/eval/golden.jsonl`

## Summary

| metric | value |
|---|---|
| questions (answerable / out of scope) | 26 / 6 |
| recall@5 | 96.2% |
| MRR | 0.962 |
| refusal accuracy | 100.0% |
| citation validity | 100.0% |
| citation hit rate (answered questions citing an expected article) | 96.2% |
| errors (dependency failures) | 0 |

## Per question

| id | expected | top retrieved | best score | refused (expected) | reason | cited |
|---|---|---|---|---|---|---|
| inc-01 | income_tax_34_2014:9 | income_tax_34_2014:9, income_tax_34_2014:4, income_tax_34_2014:11 | 0.773 | no (no) | - | income_tax_34_2014:9 |
| inc-02 | income_tax_34_2014:11 | income_tax_34_2014:11, income_tax_34_2014:9, income_tax_34_2014:12 | 0.756 | no (no) | - | income_tax_34_2014:11 |
| inc-03 | income_tax_34_2014:11 | income_tax_34_2014:11, income_tax_34_2014:12, income_tax_34_2014:4 | 0.704 | no (no) | - | income_tax_34_2014:11 |
| inc-04 | income_tax_34_2014:5 | income_tax_34_2014:5, income_tax_34_2014:9, income_tax_34_2014:4 | 0.788 | no (no) | - | income_tax_34_2014:5 |
| inc-05 | income_tax_34_2014:8 | income_tax_34_2014:8, sales_tax_6_1994:53, income_tax_34_2014:66 | 0.737 | no (no) | - | income_tax_34_2014:8 |
| inc-06 | income_tax_34_2014:7 | income_tax_34_2014:7, income_tax_34_2014:6, income_tax_34_2014:9 | 0.725 | no (no) | - | income_tax_34_2014:7 |
| inc-07 | income_tax_34_2014:4 | income_tax_34_2014:4, income_tax_34_2014:9, income_tax_34_2014:11 | 0.749 | no (no) | - | income_tax_34_2014:4 |
| inc-08 | income_tax_34_2014:4 | income_tax_34_2014:4, income_tax_34_2014:9 | 0.723 | no (no) | - | income_tax_34_2014:4 |
| inc-09 | income_tax_34_2014:13 | income_tax_34_2014:13, income_tax_34_2014:14, income_tax_34_2014:18 | 0.780 | no (no) | - | income_tax_34_2014:13 |
| inc-10 | income_tax_34_2014:14 | income_tax_34_2014:14, income_tax_34_2014:15, income_tax_34_2014:13 | 0.742 | no (no) | - | income_tax_34_2014:14 |
| inc-11 | income_tax_34_2014:15 | income_tax_34_2014:15, income_tax_34_2014:2, income_tax_34_2014:14 | 0.758 | no (no) | - | income_tax_34_2014:15 |
| inc-12 | income_tax_34_2014:16 | income_tax_34_2014:16, income_tax_34_2014:20, income_tax_34_2014:2 | 0.790 | no (no) | - | income_tax_34_2014:16 |
| inc-13 | income_tax_34_2014:17 | income_tax_34_2014:17, income_tax_34_2014:26, income_tax_34_2014:75 | 0.704 | no (no) | - | income_tax_34_2014:17, income_tax_34_2014:75 |
| inc-14 | income_tax_34_2014:17 | income_tax_34_2014:17, income_tax_34_2014:60, income_tax_34_2014:13 | 0.706 | no (no) | - | income_tax_34_2014:17 |
| inc-15 | income_tax_34_2014:10 | income_tax_34_2014:10, income_tax_34_2014:11, income_tax_34_2014:4 | 0.738 | no (no) | - | income_tax_34_2014:10 |
| inc-16 | income_tax_34_2014:18 | income_tax_34_2014:18, income_tax_34_2014:36, income_tax_34_2014:19 | 0.765 | no (no) | - | income_tax_34_2014:18 |
| inc-17 | income_tax_34_2014:26 | income_tax_34_2014:26, income_tax_34_2014:17, income_tax_34_2014:59 | 0.764 | no (no) | - | income_tax_34_2014:26 |
| sal-01 | sales_tax_6_1994:6 | sales_tax_6_1994:6, sales_tax_6_1994:16, sales_tax_6_1994:17 | 0.737 | no (no) | - | sales_tax_6_1994:6 |
| sal-02 | sales_tax_6_1994:2 | sales_tax_6_1994:13, sales_tax_6_1994:8, sales_tax_6_1994:14 | 0.777 | no (no) | - | sales_tax_6_1994:13, sales_tax_6_1994:14, sales_tax_6_1994:8 |
| sal-03 | sales_tax_6_1994:13 | sales_tax_6_1994:13, sales_tax_6_1994:26, sales_tax_6_1994:4bis | 0.800 | no (no) | - | sales_tax_6_1994:13 |
| sal-04 | sales_tax_6_1994:20 | sales_tax_6_1994:20, sales_tax_6_1994:53, sales_tax_6_1994:48 | 0.719 | no (no) | - | sales_tax_6_1994:20 |
| sal-05 | sales_tax_6_1994:31 | sales_tax_6_1994:31, sales_tax_6_1994:32, sales_tax_6_1994:68 | 0.753 | no (no) | - | sales_tax_6_1994:31, sales_tax_6_1994:68 |
| sal-06 | sales_tax_6_1994:7 | sales_tax_6_1994:7, sales_tax_6_1994:10, sales_tax_6_1994:23 | 0.737 | no (no) | - | sales_tax_6_1994:7 |
| sal-07 | sales_tax_6_1994:10 | sales_tax_6_1994:10, sales_tax_6_1994:4, sales_tax_6_1994:11 | 0.740 | no (no) | - | sales_tax_6_1994:10 |
| sal-08 | sales_tax_6_1994:38 | sales_tax_6_1994:38, sales_tax_6_1994:39, sales_tax_6_1994:18 | 0.742 | no (no) | - | sales_tax_6_1994:38, sales_tax_6_1994:18 |
| sal-09 | sales_tax_6_1994:16 | sales_tax_6_1994:16, sales_tax_6_1994:8, sales_tax_6_1994:41 | 0.797 | no (no) | - | sales_tax_6_1994:16 |
| oos-01 | - | income_tax_34_2014:66, sales_tax_6_1994:31, sales_tax_6_1994:32 | 0.631 | yes (yes) | MODEL_REFUSED | - |
| oos-02 | - | income_tax_34_2014:3, income_tax_34_2014:79, income_tax_34_2014:4 | 0.561 | yes (yes) | LOW_RETRIEVAL_SCORE | - |
| oos-03 | - | income_tax_34_2014:11, sales_tax_6_1994:15, income_tax_34_2014:3 | 0.584 | yes (yes) | LOW_RETRIEVAL_SCORE | - |
| oos-04 | - | income_tax_34_2014:2, income_tax_34_2014:22, income_tax_34_2014:74 | 0.557 | yes (yes) | LOW_RETRIEVAL_SCORE | - |
| oos-05 | - | sales_tax_6_1994:74, sales_tax_6_1994:35, sales_tax_6_1994:9 | 0.513 | yes (yes) | LOW_RETRIEVAL_SCORE | - |
| oos-06 | - | income_tax_34_2014:51, sales_tax_6_1994:60, income_tax_34_2014:52 | 0.636 | yes (yes) | MODEL_REFUSED | - |
