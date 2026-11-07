# `tevatron.retriever` — changes

Notation used below:

| Symbol | Meaning |
|---|---|
| `b` | per-device batch size (`per_device_train_batch_size`) |
| `B` | global batch size = `b × world_size` (after `all_gather` in DDP) |
| `G` | `train_group_size` (1 positive + `G-1` documents per query) |
| `S` | total number of sub-questions in the local batch (varies per query) |
| `H` | hidden size |

Sub-folders have their own README:
[`modeling/`](modeling/README.md) · [`driver/`](driver/README.md) ·
[`driver/diversify/`](driver/diversify/README.md) · [`callback/`](callback/README.md)

---

## Which pieces the CoveR paper pipeline uses

| Stage | Driver | Dataset | Collator | Trainer | Validator |
|---|---|---|---|---|---|
| Pre-fine-tuning (MS-MARCO / SCOPE-flatten) | `driver/train_dev.py` | `dataset.TrainDataset` | `collator.TrainCollator` | `trainer.TevatronTrainer` | `callback/nanobeir_eval.py` |
| CoveR training (SCOPE coverage buckets) | `driver/train_dualdistil.py` | `dataset_dev.DualDistilTrainDataset` | `collator_dev.DualDistilTrainCollator` | `trainer_dev.TevatronDualDistilTrainer` | `callback/cruxmds_eval.py` |
| Encoding | `driver/encode.py` | `dataset.EncodeDataset` | `collator.EncodeCollator` | – | – |
| Search | `driver/search.py` | – | – | – | – |

Everything else listed below is experimental and not needed to reproduce the paper.

> **Multi-view query encoding** (`num_views`, `view_pooling`, view-orthogonality loss, multi-vector
> search) was removed on 2026-11-07. It is fully documented, with all its code, in
> [`MULTIVIEW.md`](MULTIVIEW.md).

---

## `arguments.py`

New fields only (all upstream fields are unchanged, except the JAX `dtype` default which changed
from `float32` to `float16`).

### `ModelArguments`

No new fields (the multi-view fields were removed; see [`MULTIVIEW.md`](MULTIVIEW.md)).

### `DataArguments`

| Field | Default | Meaning |
|---|---|---|
| `exclude_title` | `False` | Drop the `title` column from the corpus (training and encoding). |
| `request_as_query` | `False` | Replace `query_text` with `request_text` in the training data (use the long research request as the query). |
| `train_irrelevant_size` | `0` | Number of "irrelevant" documents sampled per query in `WideDistilTrainDataset` (taken out of the `G-1` negative slots). |
| `subquery_prefix` | `''` | Prefix added to every sub-question (e.g. `"search_query: "`). |

### `TevatronTrainingArguments` — loss weights

| Field | Default | Used by | Meaning |
|---|---|---|---|
| `contrastive_lambda` | `1.0` | DualDistil | Weight of the query→document InfoNCE (`L_q-ce`). |
| `sq_contrastive_lambda` | `0.0` | DualDistil | Weight of the sub-question→document InfoNCE (`L_sq-ce`). |
| `covdistil_lambda` | `0.0` | DualDistil | Weight of the **coverage distillation** loss (sub-question teacher → query student; always KL divergence). |

### `save_args_to_json(output_path, **arg_groups)`

- **Input:** a path, and any number of dataclass instances as keyword arguments (e.g. `model_args=…, data_args=…`).
- **Output:** writes `{group_name: asdict(args)}` as JSON (rank 0 only; creates the directory if needed). `train_dev.py` and `train_dualdistil.py` use it to save `<output_dir>/arguments.json`.

---

## `dataset.py` (modified upstream file)

### `TrainDataset`

Changes:

- `exclude_title` removes the corpus `title` column; otherwise the title is prepended to the text
  when it has more than 2 characters (Tevatron corpora use `"-"` as an empty title).
- `request_as_query` maps `request_text → query_text`.
- **The legacy upstream format (`positive_passages` / `negative_passages` with inline text) was
  removed.** Only the ID-based format below is supported.
- Works without a trainer attached (epoch and seed fall back to 0).

**Input row:**

```json
{"query_id": "...", "query_text": "...",
 "positive_document_ids": ["d1", ...], "negative_document_ids": ["d7", ...]}
```

plus a corpus with `docid`, `text` and optionally `title`.

**`__getitem__` output:** `(query, documents)` where

- `query = (query_prefix + query_text, image, video, audio)`
- `documents` = `G` tuples `(passage_prefix + text, image, video, audio)`. The first is the
  positive; the rest are negatives, sampled deterministically per epoch.

### `EncodeDataset`

Accepts more column names, so BEIR, CRUX and NeuCLIR files can be encoded without conversion:

| | ID column (first match wins) | Text column (first match wins) |
|---|---|---|
| query (`encode_is_query=True`) | `query_id`, `id`, `request_id`, `qid` | `query_text`, `query`, `request`, `problem_statement` |
| passage | `docid`, `id`, `_id` | `text`, `contents` (title prepended if present) |

`exclude_title` drops the title column before encoding.

**`__getitem__` output:** `(content_id, prefix + text, image, video, audio)`.

### `DistilTrainDataset`

Now reads the teacher scores **from the training row** instead of from a separate lookup.

**Input row:** as `TrainDataset`, plus `positive_document_scores` and `negative_document_scores`
(aligned with the ID lists).

**Output:** `(query, documents, scores)`, where `scores` is a list of `G` floats aligned with
`documents`.


### `WideDistilTrainDataset` (new) — base of the CoveR dataset

Same as `DistilTrainDataset`, but the `G` documents per query are filled from three pools:

1. **1 positive** from `positive_document_ids` / `positive_document_scores`
2. **`train_irrelevant_size` irrelevant docs** from `irrelevant_document_ids` / `irrelevant_document_scores`
3. **the remaining `G-1-train_irrelevant_size` negatives** from `negative_document_ids` / `negative_document_scores`

Each pool is shuffled with a per-item seed and rotated by epoch; if a pool is smaller than needed,
it samples with replacement.

**Output:** `(query, documents, scores)`, with `len(documents) == len(scores) == G`.

> With the SCOPE splits (e.g. `pos_half.neu_low.neg_zero`) the "scores" are the coverage labels,
> i.e. how many sub-questions the document can answer.

---

## `dataset_dev.py` (new)

### `DualDistilTrainDataset(WideDistilTrainDataset)` — **used by CoveR**

**Input row:** everything `WideDistilTrainDataset` needs, plus `subquestions: List[str]`.

**`__getitem__` output:** `(query, documents, subqueries, scores)`

| Element | Type | Content |
|---|---|---|
| `query` | `(str, image)` | `query_prefix + query_text` |
| `documents` | list of `G` `(str, image, video, audio)` | positive, irrelevant and negative documents |
| `subqueries` | `List[str]` | `subquery_prefix + sq` for every sub-question (count varies per query) |
| `scores` | `List[float]` of length `G` | coverage / oracle scores aligned with `documents` |

---

## `collator.py` (modified upstream file)

Only cosmetic changes (a comment, and the field order of `DistilTrainCollator` is now
`data_args, tokenizer`). For reference, the collators used by the pipeline:

| Collator | Input features | Output |
|---|---|---|
| `TrainCollator` | `[(query, documents), …]` | `(q_batch, d_batch)`: tokenised, padded `BatchEncoding`s with `b` and `b·G` rows |
| `DistilTrainCollator` | `[(query, documents, scores), …]` | `(q_batch, d_batch, scores)`, where `scores` is a `[b, G]` tensor |
| `EncodeCollator` | `[(id, text, image), …]` | `(ids, batch)` |

---

## `collator_dev.py` (new)

The collator tokenises queries, sub-questions and documents separately. Each sequence is truncated
to `query_max_len` / `passage_max_len`, optionally gets an EOS token (`append_eos_token`), and is
padded to the longest sequence in the batch (`pad_to_multiple_of` is respected).

### `DualDistilTrainCollator` — **used by CoveR**

- **Input:** `[(query, documents, subqueries, scores), …]` from `DualDistilTrainDataset`.
- **Output:** `(q_batch, d_batch, sq_batch, num_subqueries, reranker_scores)`

| Output | Shape | Note |
|---|---|---|
| `q_batch` | `input_ids [b, Lq]` | queries |
| `d_batch` | `input_ids [b·G, Ld]` | documents, grouped per query |
| `sq_batch` | `input_ids [S, Lq]` | every sub-question of the batch, flattened in query order |
| `num_subqueries` | `List[int]`, length `b` | number of sub-questions per query, used to un-flatten `sq_batch` |
| `reranker_scores` | `Tensor [b, G]` (`torch_dtype`, default bf16) | coverage labels |

---

## `trainer.py` (modified upstream file)

### `TevatronTrainer`

In-training evaluation is done by a **validator** (see [`callback/`](callback/README.md)), not by
HF's loss-based `evaluate()`. No driver passes an `eval_dataset`, and there is **no evaluation
loss**: the validator reports ranking metrics on separate validation sets.

- `set_validator(validator)`: attaches a validator.
- `training_step`: when a validator is attached and `global_step % eval_steps == 0` (including
  step 0), it calls `validation_step(model)` before the normal step. If `--eval_steps` is unset, this is skipped.
- `validation_step(model)` (new): on rank 0 only, runs `validator.run(model, device)` and logs every
  metric as `eval/<name>` and their mean as `eval/avg`. The other ranks go straight on and wait at the
  next synchronisation. HF's own `prediction_step` is **not** overridden.
- `compute_loss`: unchanged from upstream (`model(query, passage).loss`).
- `autocast_smart_context_manager`: forces bf16/fp16 autocast based on `--bf16`/`--fp16`.

> Do not pass `--eval_strategy steps`: HF would then run its own `evaluate()`, which needs an
> `eval_dataset`. `--eval_steps N` alone is what sets the validator interval.

### `DistilTevatronTrainer`

Same KL-distillation loss and `training_step` as upstream. The only change is that it unwraps
`model.module` (DDP / accelerate) before gathering `reranker_labels`. The validator hook is
inherited from `TevatronTrainer`.

**Loss:** `KL( softmax(teacher/T) ‖ log_softmax(student_local/T) )`, where `student_local` is
the `[B, G]` block of each query's own documents and `T = distil_temperature`.

---

## `trainer_dev.py` (new) — training objectives

`TevatronDualDistilTrainer` computes a **student** score matrix and a **sub-question teacher**
score matrix, then combines several losses. (Removed on 2026-11-07: the earlier `TevatronCovDistilTrainer`,
which this trainer fully covers, and the oracle score-distillation terms `L_q-kld` and `L_sq-kld`
with their `use_kld` / `use_crossentropy` switches, which were always off.)

Shared computation (per step):

1. **Student:** `output = model(query, passage)` gives
   - `student_scores = q_reps · p_repsᵀ` with shape `[B, B·G]` (gathered across GPUs; in-batch negatives included)
   - `loss_contrastive_stu`: the standard InfoNCE over those scores, at the model `temperature`
2. **Teacher (sub-questions):** `sq_reps = encode_query(sq_batch)` has shape `[S, H]`.
   `sq_reps · p_repsᵀ` has shape `[S, B·G]`. The rows are **averaged over each query's own
   sub-questions** (using `num_subqueries`) to give `[b, B·G]`, which is gathered to
   `teacher_scores [B, B·G]`.
3. Every 100 steps it logs diagnostics: top-10 overlap between student and teacher, positive-vs-
   hardest-negative margins of both.

### `TevatronDualDistilTrainer` — **used by CoveR** (`driver/train_dualdistil.py`)

Input batch: `(query, passage, subquery, num_subqueries, coverage_scores)`. The coverage scores
are **not used by the loss** (they are still produced by the dataset and collator).

| Log name | Loss | Definition |
|---|---|---|
| `rel-constrast` | `L_q-ce` | InfoNCE of query vs. documents (positive = index `i·G`) |
| `subrel-constrast` | `L_sq-ce` | InfoNCE on `teacher_scores / temperature` |
| `cov-distil` | `L_cov` | KL( softmax(`teacher_scores.detach()`/T) ‖ log_softmax(`student_scores`/T) ) over the full `[B, B·G]` matrix (**self coverage distillation**) |

```
loss = contrastive_lambda    * L_q-ce
     + sq_contrastive_lambda * L_sq-ce
     + covdistil_lambda      * L_cov
```

`T = distil_temperature`. The teacher is the same encoder, so `L_cov` makes the request
embedding imitate the average of its sub-question embeddings; the teacher side is detached.

Settings used in the paper scripts (`slurm/*.sh`):

| Script | `contrastive_lambda` | `sq_contrastive_lambda` | `covdistil_lambda` | Effective loss |
|---|---|---|---|---|
| `relevance-*-pft.scope-{5k,10k}.sh` | 1.0 | 0.0 | 0.1 | `L_q-ce + 0.1·L_cov` |
| `unsupervised.scope-{5k,10k}.sh` | 1.0 | 0.0 | 0.0 | `L_q-ce` only |
| `exp-for-init/*.sh` | 1.0 | 1.0 | 0.0 | `L_q-ce + L_sq-ce` |
| `training/ce-contrastive.self.pft.sh` | 1.0 | 1.0 | `$lambdaA` | `L_q-ce + L_sq-ce + λ·L_cov` |

> Known issue: `teacher_scores` is only defined inside the `is_ddp` branch, so single-process
> (non-distributed) training raises an `UnboundLocalError`.

---

## Other new files

| File | Status | What it is |
|---|---|---|
| `compare.py` | reference only | A snapshot of `dataset.py` that still contains the legacy `positive_passages` code path. Not imported anywhere. |
| `unittest.py` | broken scratch | Imports a non-existent `dataset_mixed` module. Not a real test. |
