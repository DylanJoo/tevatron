# `tevatron.retriever.modeling` — changes in this fork (CoveR)

Changes relative to upstream Tevatron `4b1f32d`.

`dense.py`, `splade.py` and `unicoil.py` are **identical to upstream**. The CoveR models use
`DenseModel` with `--pooling mean --normalize`, i.e. one mean-pooled, L2-normalised vector per
query, sub-question and document.

> Multi-view query encoding (`num_views`, `view_pooling`, multi-vector similarity) was removed on
> 2026-11-07. Its design and full code are archived in [`../MULTIVIEW.md`](../MULTIVIEW.md).

Notation: `B` = batch size, `G` = `train_group_size`, `H` = hidden size.

---

## `encoder.py` — `EncoderModel`

Remaining differences from upstream:

### `forward(query=None, passage=None) -> EncoderOutput`

| Mode | Input | Output |
|---|---|---|
| inference (only one of `query`/`passage`) | tokenised batch | `q_reps` `[B, H]` or `p_reps` `[B, H]` (unchanged) |
| training (`model.training`) | `query` `[B, Lq]`, `passage` `[B·G, Ld]` | unchanged: reps are `all_gather`ed in DDP (cross-device negatives); `scores` `[B, B·G]`; `loss` = InfoNCE(`scores / temperature`, target = `i·G`) |
| eval (`model.eval()` with both inputs) | same | **changed:** also computes `loss` and in-batch top-1 accuracy, returned in `logs={'acc': …}` (upstream returns `loss=None`) |

`EncoderOutput` gained a `logs: Optional[Dict]` field for this.

Nothing in the current pipeline reaches this branch: the in-training validators in
[`../callback/`](../callback/README.md) call the model with only a query or only a passage
(inference mode). It is kept for direct use, e.g. computing in-batch accuracy on a held-out split.

### `build(...)` / `load(...)`

Unchanged from upstream.

### `gradient_checkpointing_enable(**kwargs)`

**Changed:** it uses **non-reentrant** checkpointing (`use_reentrant=False`) by default and forwards
`--gradient_checkpointing_kwargs` (upstream ignores the kwargs, so HF's reentrant default was used).
Reentrant checkpointing runs a nested backward per checkpointed block, which breaks DDP when the
encoder is called several times per step, as `TevatronDualDistilTrainer` does for queries, passages and
sub-questions (`RuntimeError: Expected to mark a variable ready only once`). Only relevant with
`--gradient_checkpointing`.
