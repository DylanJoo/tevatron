# `tevatron.retriever.driver.diversify` — new in this fork (experimental)

Search-time diversification baselines. These are drop-in replacements for `driver/search.py`
that read the same encoded pickles. They are **not used by the CoveR paper scripts**.

Common inputs (same as `search.py`):

| Flag | Meaning |
|---|---|
| `--query_reps` | pickle `(q_reps [N, H], q_lookup: List[qid])` from `driver/encode.py` |
| `--passage_reps` | glob of pickles `(p_reps [M, H], p_lookup: List[docid])` |
| `--depth` | number of candidates to retrieve and return (default 1000) |
| `--batch_size` | FAISS search batch size (default 128) |
| `--save_ranking_to` | output path |
| `--save_text` | write `qid \t docid \t score` lines; otherwise a pickle of `(scores, docids)` |

Both scripts build a flat inner-product FAISS index and move it to the GPU(s) when available.

---

## `mmr.py` — Maximal Marginal Relevance re-ranking

```
python -m tevatron.retriever.driver.diversify.mmr --query_reps q.pkl --passage_reps 'corpus.*.pkl' \
    --depth 100 --lambda_param 0.9 --save_ranking_to run.txt --save_text
```

1. `search_queries`: dense retrieval of the top `depth` docs per query.
2. `search_queries_mmr`: greedily re-orders those candidates. At each step it picks the doc with the highest
   `λ · score(q, d) − (1 − λ) · max_{d' ∈ selected} sim(d, d')`, where `sim` is the
   inner product of the passage embeddings and `λ = --lambda_param` (default 0.9).
3. **Output:** per query, the `depth` candidates in MMR order. The written score is the MMR score, not the dense score.

Cost is `O(depth²)` per query in pure Python, so keep `--depth` small (e.g. ≤ 100).

## `mqr.py` — Multi-query retrieval with rank fusion

Used when each query is expanded into several sub-queries (e.g. sub-questions) that were encoded
as separate rows.

**Input convention:** the IDs in `q_lookup` must be `"<qid>:::<subquery_id>"`. Everything before
`:::` is the parent query ID used for fusion.

```
python -m tevatron.retriever.driver.diversify.mqr --query_reps subq.pkl --passage_reps 'corpus.*.pkl' \
    --depth 100 --fusion_method ranklist-reciprocal-fusion --save_ranking_to run.txt --save_text
```

1. Retrieves the top `depth` docs for **every sub-query**.
2. Fuses them back to one ranking per parent `qid` according to `--fusion_method`:

| `--fusion_method` | Fused score of a document |
|---|---|
| *(unset)* | no fusion; returns per-sub-query rankings with the original IDs |
| `score-sum` | sum of its dense scores over all sub-queries |
| `ranklist-round-robin` | interleave the sub-query rankings, taking one new doc per sub-query per round; score = `1/(position+1)` |
| `ranklist-reciprocal-fusion` | `Σ 1/(rank+1)` over sub-queries (RRF with `k=0`) |

3. **Output:** one ranking per parent `qid`.
