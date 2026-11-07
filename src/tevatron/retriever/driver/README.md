# `tevatron.retriever.driver` — changes in this fork (CoveR)

Entry points run as `python -m tevatron.retriever.driver.<name>` (training scripts are usually
launched through `accelerate launch` or `torchrun`). Changes relative to upstream Tevatron `4b1f32d`.
Unchanged upstream drivers (`train_distil.py`, `train_mm.py`, `encode_mm.py`, `jax_*.py`,
`vllm_*.py`) are not described here.

All training drivers parse `ModelArguments`, `DataArguments` and `TevatronTrainingArguments` from
[`../arguments.py`](../README.md#argumentspy) (CLI flags, or a single `.json` file).

---

## Overview

| Driver | Status | Dataset → Collator → Trainer | Validator |
|---|---|---|---|
| `train_dualdistil.py` | **CoveR training** | `DualDistilTrainDataset` → `DualDistilTrainCollator` → `TevatronDualDistilTrainer` | CRUX-MDS (`callback/cruxmds_eval.py`) |
| `train_dev.py` | **Pre-fine-tuning (PFT)** | `TrainDataset` → `TrainCollator` → `TevatronTrainer` | NanoBEIR (`callback/nanobeir_eval.py`) |
| `encode.py` | **Inference** (modified) | `EncodeDataset` → `EncodeCollator` → `DenseModel.load` | – |
| `search.py` | **Inference** (modified) | – | – |
| `train_distil_dev.py` | experimental, **broken** | `DistilTrainDataset` → `DistilTrainCollator` → `DistilTevatronTrainer` | imports `callback.crux_eval`, which does not exist |
| `train.py` | upstream (unchanged) | `TrainDataset` → `TrainCollator` → `TevatronTrainer`/`GradCacheTrainer` | – |
| `encode_lsr.py` | experimental | standalone SPLADE-style sparse encoder | – |
| `diversify/` | experimental | see [`diversify/README.md`](diversify/README.md) | – |

The objectives of each trainer are documented in [`../README.md`](../README.md#trainer_devpy-new--training-objectives).

---

## `train_dualdistil.py` — CoveR training

1. Writes `<output_dir>/arguments.json` with `save_args_to_json`.
2. Builds the tokenizer, then sets the pad token (EOS if missing) and the padding side.
3. `DenseModel.build(model_args, training_args, torch_dtype=…)`.
4. `DualDistilTrainDataset(data_args)` + `DualDistilTrainCollator(data_args, tokenizer)`.
5. `TevatronDualDistilTrainer` with a CRUX-MDS `Validator(EncodeCollator(...), 512)`, run every `eval_steps`.
6. Resumes from the last checkpoint in `output_dir` if one exists; saves the model and tokenizer at the end.

**Data it expects** (`--dataset_name`, `--dataset_split`, `--corpus_name`), e.g.
`DylanJHJ/crux-researchy-kdnew-ext` split `pos_half.neu_low.neg_zero` with corpus
`DylanJHJ/crux-researchy-corpus`. Each row needs:

```
query_id, query_text (or request_text with --request_as_query),
positive_document_ids, positive_document_scores,
negative_document_ids, negative_document_scores,
irrelevant_document_ids, irrelevant_document_scores,
subquestions
```

Key flags: `--request_as_query`, `--subquery_prefix`, `--train_group_size`,
`--contrastive_lambda`, `--sq_contrastive_lambda`, `--covdistil_lambda`, `--distil_temperature`,
`--temperature`.

This driver replaces the removed `train_covdistil.py`. Its loss is identical
(`contrastive_lambda·L_q-ce + sq_contrastive_lambda·L_sq-ce + covdistil_lambda·L_cov`).

**Output:** a HF checkpoint in `output_dir` (encoder weights + tokenizer), periodic
`checkpoint-*`, and `arguments.json`.

## `train_dev.py` — pre-fine-tuning

Same flow as above, but with standard contrastive training (`TrainDataset`/`TrainCollator`/
`TevatronTrainer`) and the NanoBEIR validator. Used for the MS-MARCO
(`Tevatron/msmarco-passage-new`) and SCOPE-flatten (`flatten.pos_5.neg_1`) PFT models.
It does not write `arguments.json`. A `MultiTrainDataset` path is left commented out.

**Data it expects:** `query_id, query_text, positive_document_ids, negative_document_ids` +
a corpus with `docid, text[, title]`.

## `encode.py` — modified

Changes:

- If `--dataset_name` / `--dataset_split` were given as lists, it uses the first element.
- Prints the model and data arguments at start-up.

**Input:** `--dataset_name`/`--dataset_path` (+ `--encode_is_query` for queries). Accepted
ID/text column names are listed under `EncodeDataset` in [`../README.md`](../README.md#encodedataset).

**Output:** `--encode_output_path`, a pickle of `(reps: np.ndarray [N, H], lookup: List[id])`.

## `search.py` — upstream (unchanged)

**Input:** `--query_reps` (one pickle), `--passage_reps` (glob of pickles), `--depth`, `--batch_size`.

**Output:** `--save_ranking_to`, either plain text `qid \t docid \t score` (`--save_text`) or a
pickle of `(scores, indices)`. Convert the text file to TREC with
`python -m tevatron.utils.format.convert_result_to_trec`.

## `train.py` — upstream (unchanged)

Restored to upstream on 2026-11-07 (the fork's `QrelDataset` eval was removed). Only
`slurm/modernbert/train.modernbert.multinode.sh` calls it.

## `encode_lsr.py` — experimental sparse (SPLADE-style) encoder

- **Model:** `BertForMaskedLM`; doc vector = `max_over_tokens( log(1 + ReLU(MLM logits)) )` → vocabulary-sized weights.
- **Input:** `--collection` (a `.jsonl` file or folder of them, rows `{id, contents}`) or
  `--dataset_name/--dataset_split` (HF corpus with `docid, text[, title]`). Flags: `--exclude_title`, `--max_length`, `--batch_size`, `--device`.
- **Output:** `<collection_output>_<file>_<shard>` JSONL with `{"id", "contents", "vector": {token: int weight}}`,
  where weights are multiplied by `--quantization_factor` and filtered by `--minimum`. This is the
  Anserini/Pyserini `JsonVectorCollection` format. It processes 1M documents per shard.
