# `tevatron.retriever.callback` — new in this fork

Light-weight **re-ranking validators** that run inside training. A driver attaches one with
`trainer.set_validator(Validator(EncodeCollator(data_args, tokenizer), 512))`. Then
`TevatronTrainer.training_step` calls `validation_step` → `validator.run(model, device)` every
`--eval_steps` steps (rank 0 only). Each returned metric is logged as `eval/<name>`, and their mean as `eval/avg`.

Both validators have the same interface:

| Method | Input | Output |
|---|---|---|
| `Validator(collator, batch_size)` | an `EncodeCollator`; `batch_size` is ignored (fixed to 512) | – |
| `run(model, device)` | the (unwrapped or DDP) `EncoderModel` and a device | `Dict[str, float]` of metrics |

How `run` works for each split of the validation set:

1. Encode **all queries of the split in one batch** (`collator.encode_is_query = True`).
2. For each query, encode **only its candidate documents** (`document_ids` in the validation
   row; a fixed re-ranking pool, not a full-corpus search) and score them with `q · dᵀ`.
3. Build `run = {qid: {docid: score}}` and evaluate it.

> The validators flip `collator.encode_is_query` but do **not** change the query or passage
> prefix. They use whatever `query_prefix`/`passage_prefix` the shared `DataArguments` hold
> (in the paper scripts both are set, so this is fine).

---

## `cruxmds_eval.py` — CRUX-MDS (used by `train_dualdistil.py`)

- **Validation queries:** HF `DylanJHJ/valid-crux-mds`. Every split has rows `{query_id, query_text, document_ids}`.
- **Corpus:** HF `DylanJHJ/crux-mds-corpus`. The `train` and `test` splits are merged; rows `{id, contents}`.
- **Qrels and judgements:** read from `$CRUX_ROOT/crux-mds-<split>/qrels/div_qrels-tau3.txt` and
  `$CRUX_ROOT/crux-mds-<split>/judge`. **`CRUX_ROOT` must be set.**
- **Metric:** `crux.evaluation.rac_eval(run, qrel, div_qrel, tau=3, cutoff=10, judge=ratings, filter_by_oracle=True)`.
- **Output:** `{"<split>-<metric>": mean_value, …}` for every metric `rac_eval` returns (coverage / diversity at 10).
- **Requires:** the [`crux`](https://github.com/DylanJHJ/crux) package.

## `nanobeir_eval.py` — NanoBEIR (used by `train_dev.py`)

- **Validation queries:** HF `DylanJHJ/valid-nano-beir`. Split names look like `nano_beir_scifact` etc.; rows `{query_id, query_text, document_ids}`.
- **Corpus:** HF `DylanJHJ/beir-subset-corpus`. The split name is the validation split without `nano_`; rows `{docid, text}`.
- **Qrels:** `ir_datasets.load(<split with '_'→'-' and '.'→'/'>).qrels_dict()`.
- **Metric:** `nDCG@10` via `ir_measures`.
- **Output:** `{"<split>": nDCG@10, …}`.
