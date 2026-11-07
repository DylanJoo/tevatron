# ColBERT indexing and search — new in this fork

Wrappers around the [ColBERT](https://github.com/stanford-futuredata/ColBERT) PLAID index that read
queries and documents through Tevatron's `EncodeDataset`. That means the same `--dataset_name`,
`--dataset_split`, `--passage_prefix`/`--query_prefix` and ID/text column fallbacks as
`tevatron.retriever.driver.encode` all work. They are **not used by the CoveR paper scripts**.

Requires the `colbert-ai` package. Both scripts parse Tevatron's
`ModelArguments, DataArguments, TevatronTrainingArguments` plus their own arguments, either from
CLI flags or from a single `.json` file.

---

## `index.py` — build a ColBERT index

**Input:** a corpus loaded with `EncodeDataset` (`--encode_is_query` must be **off**), and
`--model_name_or_path` (a ColBERT checkpoint; a `.dnn` file means a v1 checkpoint, which then
needs `--base_model`).

**Steps** (`--step`). Run them in order, possibly as separate jobs:

| Step | What it does |
|---|---|
| `prepare` / `prepare-gpu` / `prepare-cpu` | sample tokens and train the k-means centroids (on GPU or CPU) |
| `encode` | encode all passages and compress the residuals with `--nbits` bits per dimension |
| `finalize` | build the IVF and write the final index |

Important arguments (`ColBERTArguments`): `--nbits` (default 2), `--n_gpus`, `--index_name`,
`--experiment`, `--root`, `--max_num_partitions`, `--max_sampled_pid`,
`--lazy_collection_loader`, `--ivf_num_processes`, `--ivf_use_tempdir`, `--ivf_merging_ways`,
`--use_lagacy_build_ivf`, `--reuse_centroids_from`.

**Output:**

- the ColBERT index at `dirname(--encode_output_path)/<index_name>`, where `index_name` defaults
  to `basename(--encode_output_path)`
- `<index_name>.docid_map.pkl`: a list that maps ColBERT's integer passage id to the original docid.

## `search.py` — search a ColBERT index

**Input:** queries loaded with `EncodeDataset` (**`--encode_is_query` is required**), plus
`SearchArguments`:

| Argument | Meaning |
|---|---|
| `--index_path` | the `--encode_output_path` used at indexing time (also locates `docid_map.pkl`) |
| `--run_path` | output file |
| `--depth` | results per query |
| `--nbits` | must match the index |
| `--only_approx` | approximate (centroid-only) scoring: faster, less accurate |
| `--no_gpu_search` | run the FAISS search on CPU to avoid GPU OOM (encoding may still use the GPU) |

**Output:** `--run_path` with lines `qid \t docid \t score`. Passage-level hits are mapped to
docids with `docid_map.pkl` and reduced with **MaxP** (`maxp()`: the best passage score per docid),
sorted by score. Convert to TREC format with `python -m tevatron.utils.format.convert_result_to_trec`.
