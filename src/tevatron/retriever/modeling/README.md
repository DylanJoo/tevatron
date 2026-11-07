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

The eval branch only runs through `TevatronTrainer.compute_loss`'s legacy dict-input path. The
in-training evaluation the drivers use is the validator in [`../callback/`](../callback/README.md),
which calls the model in inference mode, so this branch is effectively unused.

### `build(...)` / `load(...)`

Unchanged except that they `print` the `hf_kwargs` passed to `from_pretrained` (debug output).
