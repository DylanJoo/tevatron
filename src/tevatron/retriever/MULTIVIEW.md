# Multi-view query encoding (removed, archived here)

**Status:** removed from the code base on 2026-11-07 to stay close to upstream Tevatron.
The CoveR paper models never used it (`num_views=0` in every paper run).

**Last commit that contains the full implementation:** `1130564` on branch `2026-Oct`
(also `72b5e92`, 2026-01-18, "update searching with multiview"). The history is:

| Commit | Date | What was added |
|---|---|---|
| `93e3e67` | 2026-01-05 | multi-view reps in the model, trainer and search |
| `ef42ea3` | 2026-01-06 | `cluster` view pooling |
| `72b5e92` | 2026-01-18 | multi-vector search with score aggregation |

To bring it back, see [How to restore](#how-to-restore). All the removed code is also reproduced
verbatim below, so this file is self-contained.

---

## 1. Idea

A research request covers several sub-topics, but a single query vector has to compress all of
them into one point. Multi-view encoding gives the query **`V` vectors ("views")**, each meant to
capture a different aspect. A document then only needs to match **one or several** views instead of
the whole request:

```
query  "search_query:[unused0][unused1][unused2][unused3][unused4] <request text>"
                     └──────── V = 5 view tokens ────────┘
encoder → hidden states [B, L, H]
        → V view vectors [B, V, H]   (one per view token, see pooling modes)
score(q, d) = aggregate_v( view_v · d )      (mean or max over views)
```

Documents and sub-questions stay single-vector. This fits CoveR naturally: the sub-question
teacher already gives several "aspect" vectors per request, and the views can learn to spread over
those aspects. The orthogonality loss (§4) pushes the views apart so they don't collapse into one.

---

## 2. Settings

| Argument | Group | Default | Meaning |
|---|---|---|---|
| `--num_views` | `ModelArguments` | `0` | Number of view tokens `V`. `0` disables multi-view (standard pooling). |
| `--view_pooling` | `ModelArguments` | `'independent'` | How each view vector is built: `independent`, `cumulative`, `cluster`, `weighted` (§3). |
| `--view_start_idx` | `ModelArguments` | `5` | Token position of the first view token. With a BERT-style tokenizer and prefix `search_query:` the positions are `0:[CLS] 1:search 2:_ 3:query 4:: 5:[unused0]`. **Check this for your tokenizer.** |
| `--aggregation_strategy` | `TevatronTrainingArguments` | `'sum'` | How `[B, V, N]` view scores are reduced to `[B, N]` during training: `max`, otherwise mean. Note that the model and `load()` defaulted to `'mean'`; only `max` vs. everything-else mattered. |
| `--view_orthogonalize_method` | `TevatronTrainingArguments` | `None` | `mse`, `abs`, or anything else (e.g. `pairwise`) for the squared off-diagonal (§4). |
| `--view_orthogonalize_lambda` | `TevatronTrainingArguments` | `0.0` | Weight of the orthogonality loss. |
| `--aggregation_strategy` (search) | `driver/search.py` CLI | `'mean'` | Fusion of per-view retrieval lists: `sum`, `mean`, `max`, `rrf` (§5). |

**The view tokens must be in the query text.** There was no code that inserted them. They were
added through the query prefix, e.g.

```
--query_prefix "search_query:[unused0][unused1][unused2][unused3][unused4]"
```

and `--num_views` / `--view_start_idx` must match that prefix. Check that each `[unusedK]` is a
**single** token in your tokenizer, or add it as a special token. If it isn't, the view
positions shift and pool the wrong tokens.

---

## 3. View pooling modes (`DenseModel.encode_query`)

Notation: `h` = last hidden states `[B, L, H]` (padding masked to 0), view tokens at positions
`s … s+V-1` with `s = view_start_idx`. "Query tokens" are all non-padding tokens except the view tokens.
Every mode ends with L2 normalisation per view, giving `[B, V, H]`.

| Mode | View `v` | Intuition |
|---|---|---|
| `independent` (default) | `mean(query tokens) + h[s+v]` | shared request meaning + a view-specific offset |
| `cumulative` | `(Σ query tokens + Σ_{u≤v} h[s+u]) / (n_query + v + 1)` | each view adds one more view token, coarse to fine |
| `cluster` | hard-assign every query token to one view with Gumbel-softmax over `h[s+v] · h[token]`, then mean of the assigned tokens | each view owns a disjoint subset of the request tokens (aspect segmentation) |
| `weighted` | `softmax_tokens(h[s+v] · h[token]) @ h` | each view is an attention-pooled summary (soft version of `cluster`) |

Known issues in the old code:

- `cumulative` masked positions `5:5+V` with a hard-coded `5` instead of `view_start_idx`.
- `cluster` uses `gumbel_softmax(hard=True)` at **inference too**, so encoding is stochastic.
  Use `argmax` (or `hard=False` with a low temperature) at eval time.
- In `cluster` and `weighted`, `query_reps` still contains the padding positions (zeroed), which
  take part in the similarity and softmax.
- **`weighted` is broken.** The softmax output is `[B, V, L−V]` (view tokens excluded), but it is
  multiplied with `masked_hiddens`, which is `[B, L, H]`, so the matmul raises a shape error. It
  should use `query_reps`, as `cluster` does. It was probably never trained.

Removed code (`modeling/dense.py`):

```python
def encode_query(self, qry, num_views=0, view_pooling=None, view_start_idx=5):
    query_hidden_states = self.encoder(**qry, return_dict=True)
    query_hidden_states = query_hidden_states.last_hidden_state
    if num_views == 0:
        return self._pooling(query_hidden_states, qry['attention_mask'])

    masked_hiddens = query_hidden_states.masked_fill(~qry['attention_mask'][..., None].bool(), 0.0)
    query_mask = qry['attention_mask'].clone()

    if view_pooling=='cumulative':
        ### NOTE: cumulative/incremental pooling
        query_mask[:, 5:(5+num_views)] = False # exclude view tokens
        query_sum = (masked_hiddens * query_mask[..., None]).sum(dim=1)
        query_len = query_mask.sum(dim=1)

        views_cumsum = (masked_hiddens[:, view_start_idx:(view_start_idx+num_views), :]).cumsum(dim=1) # B 5 H
        views_len = query_len[..., None] + torch.arange(1, num_views + 1, device=masked_hiddens.device)[None, ...] # B 5
        views_reps = query_sum.unsqueeze(1) + views_cumsum # B 5 H
        reps = views_reps / views_len.unsqueeze(-1)

    elif view_pooling == 'cluster':
        ### NOTE: cluster pooling
        cluster_reps = masked_hiddens[:, view_start_idx:(view_start_idx+num_views), :] # B N H
        query_reps = torch.cat([ masked_hiddens[:, :view_start_idx, :], masked_hiddens[:, (view_start_idx+num_views):, :] ], dim=1)  # B Q H
        cluster_logit = torch.matmul(cluster_reps, query_reps.transpose(1, 2)) # B N H, B H Q = B N Q

        # each query tokens belong to a view, and then take the mean average for each subset of views
        cluster_indices = torch.nn.functional.gumbel_softmax(cluster_logit, dim=1, hard=True) # B N Q
        cluster_reps = torch.matmul(cluster_indices, query_reps) # B N Q, B Q H = B N H
        cluster_counts = (cluster_indices.sum(dim=-1) + 1e-4) # B 5
        reps = cluster_reps / cluster_counts.unsqueeze(-1)

    elif view_pooling == 'weighted':
        cluster_reps = masked_hiddens[:, view_start_idx:(view_start_idx+num_views), :] # B N H
        query_reps = torch.cat([ masked_hiddens[:, :view_start_idx, :], masked_hiddens[:, (view_start_idx+num_views):, :] ], dim=1)  # B Q H
        cluster_logit = torch.matmul(cluster_reps, query_reps.transpose(1, 2)) # B N H, B H Q = B N Q

        # each view has a distribution of query token (sum to 1), and then take the sum for each distribution to form a view.
        cluster_dist = torch.nn.functional.softmax(cluster_logit, dim=-1) # B 5 Q # along the query dimension
        reps = torch.matmul(cluster_dist, masked_hiddens) # B 5 Q, B Q H = B 5 H

    else:
        ### NOTE: independent poolign
        query_mask[:, view_start_idx:(view_start_idx+num_views)] = False # exclude view tokens
        query_sum = (masked_hiddens * query_mask[..., None]).sum(dim=1)
        query_len = query_mask.sum(dim=1)

        views_len = query_len[..., None] + torch.ones(num_views, device=masked_hiddens.device)[None, ...] # B 5
        views_reps = masked_hiddens[:, view_start_idx:(view_start_idx+num_views), :] # B 5 H

        # Option1: mean over query+special tokens
        # views_reps = query_sum.unsqueeze(1) + views_reps # B 5 H
        # reps = views_reps / views_len.unsqueeze(-1)

        # Option2: mean over query then plus special tokens
        query_reps = query_sum / query_len[..., None]
        reps = query_reps.unsqueeze(1) + views_reps # B 5 H

    reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
    return reps
```

`encode_passage(psg)` kept calling `self.encode_query(psg)` with the default `num_views=0`, so
passages were always single-vector. The trainer encoded sub-questions the same way.

---

## 4. Training

### Model plumbing (`modeling/encoder.py`)

`EncoderModel.__init__` stored four extra attributes. `build()` filled them from
`model_args` / `train_args`, and `load()` from keyword arguments:

```python
def __init__(self, encoder, pooling='cls', normalize=False, temperature=1.0,
             num_views=0, view_pooling='independent', aggregation_strategy='mean', view_start_idx=5):
    ...
    self.num_views = num_views
    self.view_pooling = view_pooling
    self.view_start_idx = view_start_idx
    self.aggregation_strategy = aggregation_strategy

def forward(self, query=None, passage=None):
    q_reps = self.encode_query(query, self.num_views, self.view_pooling, self.view_start_idx) if query else None
    ...

def compute_similarity(self, q_reps, p_reps):
    if q_reps.dim() == 2:
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
    else:
        scores_multi = torch.matmul(q_reps, p_reps.transpose(0, 1))   # [B, V, N]
        if self.aggregation_strategy=='max':
            return scores_multi.max(dim=1).values
        else:
            return scores_multi.mean(dim=1)
```

```python
# in build(...) and load(...), passed to cls(...):
num_views=model_args.num_views,
view_pooling=model_args.view_pooling,
aggregation_strategy=train_args.aggregation_strategy,
view_start_idx=model_args.view_start_idx,
```

With this, the standard InfoNCE (`L_q-ce`) and the coverage distillation (`L_cov`) worked
unchanged on the aggregated `[B, B·G]` score matrix.

### Orthogonality loss (`trainer_dev.py`)

```python
def compute_orthogonal_loss(self, views):          # views: [B, V, H] (L2-normalised)
    if views.dim() != 3:
        return torch.tensor(0)
    B, V, H = views.shape
    sim = torch.bmm(views, views.transpose(1, 2))   # [B, V, V]
    eye = torch.eye(V, device=views.device).unsqueeze(0)
    num_pairs = B * V * (V - 1)
    if self.args.view_orthogonalize_method == 'mse':
        loss = ((sim - eye) ** 2).sum() / num_pairs
    elif self.args.view_orthogonalize_method == 'abs':
        loss = torch.abs(sim * (1 - eye)).sum() / num_pairs
    else:                                            # e.g. 'pairwise'
        sim = sim * (1 - eye)
        loss = (sim ** 2).sum() / num_pairs
    return loss * self._dist_loss_scale_factor
```

In `compute_loss`:

```python
loss_orthogonal = self.compute_orthogonal_loss(q_reps)   # q_reps from model(query, passage)
...
self.log({..., "view-similarity": loss_orthogonal.item()})
loss += loss_orthogonal * self.args.view_orthogonalize_lambda
```

### Diagnostic: view-similarity heat-map (every 100 steps, W&B)

```python
import wandb
import matplotlib.pyplot as plt
from transformers.integrations import WandbCallback

if self.state.global_step % 100 == 0:
    if q_reps.dim() == 3:
        q_reps_ = q_reps.detach()
        sim_mat = torch.bmm(q_reps_, q_reps_.transpose(1, 2))  # B, V, V
        sim_mat = sim_mat.mean(dim=0).to(torch.float32).cpu().numpy()
        np.fill_diagonal(sim_mat, np.nan)

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(sim_mat, cmap="coolwarm")
        fig.colorbar(im, ax=ax)

        for cb in self.callback_handler.callbacks:
            if isinstance(cb, WandbCallback):
                control = cb.on_log(self.args, self.state, self.control, logs={"views_heatmap": wandb.Image(fig)})
        plt.close(fig)
```

### Validators (`callback/*.py`)

With multi-view queries, `query_embs[idx] @ doc_embs.T` is `[V, D]`. It was reduced with
`scores.mean(axis=0)` in `cruxmds_eval.py` and `scores.sum(axis=0)` in `nanobeir_eval.py`:

```python
if scores.ndim == 2: # only check one qid
    scores = scores.mean(axis=0)     # nanobeir: scores.sum(axis=0)
```

---

## 5. Encoding and search

`driver/encode.py` passed the view settings to `DenseModel.load`:

```python
model = DenseModel.load(
    model_args.model_name_or_path,
    pooling=model_args.pooling,
    normalize=model_args.normalize,
    lora_name_or_path=model_args.lora_name_or_path,
    num_views=model_args.num_views,
    view_pooling=model_args.view_pooling,
    cache_dir=model_args.cache_dir,
    torch_dtype=torch_dtype,
    attn_implementation=model_args.attn_implementation,
    view_start_idx=model_args.view_start_idx,
)
```

Query pickles then hold `[N, V, H]` reps; passage pickles stay `[M, H]`. Search needs the same
`--query_prefix` (with the view tokens) and `--num_views` at query-encoding time.

`driver/search.py` added `--aggregation_strategy` (default `mean`) and forwarded it:

```python
parser.add_argument('--aggregation_strategy', type=str, default='mean')
...
all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size, args.quiet, args.aggregation_strategy)
```

`searcher.py` (`FaissFlatSearcher`): with 3-D queries it searches each view separately and fuses the lists:

```python
from collections import defaultdict

def batch_search(self, q_reps, k, batch_size, quiet=False, aggregation_strategy='sum'):
    num_query = q_reps.shape[0]
    all_scores, all_indices = [], []
    for start_idx in tqdm(range(0, num_query, batch_size), disable=quiet):
        if q_reps.ndim == 2:
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
        else:
            nn_scores, nn_indices = self.parallel_search(q_reps[start_idx: start_idx + batch_size], k, aggregation_strategy)
        all_scores.append(nn_scores)
        all_indices.append(nn_indices)
    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices

def parallel_search(self, sq_reps, k, aggregation_strategy):
    """ The q_reps should be a batch of subqueries.  """
    batch_size = sq_reps.shape[0]
    num_subqueries = sq_reps.shape[1]
    sq_reps_flatten = sq_reps.reshape(-1, sq_reps.shape[-1])
    nn_scores, nn_indices = self.search(sq_reps_flatten, k)
    num_docs = nn_indices.shape[1]

    batch_indices, batch_scores = [], []
    for batch_idx in range(batch_size):
        start = batch_idx * num_subqueries
        end   = (batch_idx + 1) * num_subqueries
        batch_nn_scores  = nn_scores[start: end]
        batch_nn_indices = nn_indices[start: end]
        n = batch_nn_scores.shape[1]

        doccounts = defaultdict(int)
        doc2score = defaultdict(float)
        for i, (score, docid) in enumerate(zip(batch_nn_scores.flatten(), batch_nn_indices.flatten())):
            doccounts[docid] += 1
            if aggregation_strategy == 'max':
                doc2score[docid] = max(score, doc2score[docid])
            elif aggregation_strategy == 'rrf':
                rank = (i + 1) % n
                doc2score[docid] += 1 / (60+rank)
            else:                                   # 'sum' / 'mean'
                doc2score[docid] += score

        if aggregation_strategy == 'mean':
            for docid in doc2score:
                doc2score[docid] = doc2score[docid] / doccounts[docid]

        sorted_docs = sorted(doc2score.items(), key=lambda x: x[1], reverse=True)
        top_docs = sorted_docs[:k]
        num_docs += len(sorted_docs)
        batch_scores.append(np.array([s for _, s in top_docs]))
        batch_indices.append(np.array([d for d, _ in top_docs]))

    print('Average # retrieved documents:', num_docs / batch_size)
    return batch_scores, batch_indices
```

Known issues: the `rrf` rank is `(i + 1) % n`, so the last item of each list gets rank 0. It
should be `i % n + 1`. Also, `mean` divides by the number of views that *retrieved* the doc,
which favours docs that only one view retrieved with a high score. Training, by contrast, averages
over **all** views.

---

## 6. Experiments that used it

The scripts are kept in `~/CoveR/slurm/original/` (they were deleted from Causal-CoveR, and they call
the removed `train_covdistil` driver):

| Script | Pooling | Grid (`covdistil_lambda` × `sq_contrastive_lambda` × `view_orthogonalize_lambda`) | Split | Steps |
|---|---|---|---|---|
| `finetune.modernbert.multiview.covdistil.kld.sh` | `cluster` | `0.25` × `{0.1, 0.25}` × `{0.001, 0.0001}` | `pos_high.neg_quarter` | 7500 |
| `finetune.modernbert.multiview.covdistil.kld.cumulative_pooling.sh` | cumulative (via the old flag `--cumulative_pooling True`, which no longer exists; use `--view_pooling cumulative`) | `0.25` × `{0.25, 0.5}` × `{0.0, 0.1}` | `pos_half.neg_zero` | 5000 |
| `finetune.dev.sh` | `cluster` | `view_orthogonalize_lambda 0.1` | – | – |

Shared settings: `PRETRAINED=DylanJHJ/nomic.modernbert-base.msmarco-passage.10k`,
`--query_prefix "search_query:[unused0][unused1][unused2][unused3][unused4]" --num_views 5`,
`--request_as_query --subquery_prefix "search_query: "`, `--aggregation_strategy mean`,
`--view_orthogonalize_method pairwise`, `--covdistil_method KLD`, batch 16×4 GPUs,
`--train_group_size 8`, `--query_max_len 128`, `--passage_max_len 512`, lr `1e-4`,
dataset `DylanJHJ/crux-researchy-new`, corpus `DylanJHJ/crux-researchy-corpus`.

Model outputs went to `~/models/msmarco-passage-pft.multiview-*`. They no longer exist on disk, and no
results for them are recorded in `result/`.

---

## 7. How to restore

The old trainer (`TevatronCovDistilTrainer`) is also gone. Re-apply the pieces on top of the current
`TevatronDualDistilTrainer`:

1. `arguments.py`: add back the six arguments from §2.
2. `modeling/dense.py`: replace `encode_query` with the version in §3.
3. `modeling/encoder.py`: the `__init__` / `forward` / `compute_similarity` / `build` / `load` changes in §4.
4. `trainer_dev.py`: `compute_orthogonal_loss`, the `+ view_orthogonalize_lambda * L_orth` term and (optional) the heat-map.
5. `callback/*.py`: the `scores.ndim == 2` reduction.
6. `driver/encode.py`, `driver/search.py`, `searcher.py`: §5.

Or start from git, file by file:

```bash
cd ~/tevatron
git show 1130564:src/tevatron/retriever/modeling/dense.py   > /tmp/dense_multiview.py
git show 1130564:src/tevatron/retriever/modeling/encoder.py > /tmp/encoder_multiview.py
git show 1130564:src/tevatron/retriever/searcher.py         > /tmp/searcher_multiview.py
git diff 1130564 HEAD -- src/tevatron/retriever/            # everything removed since
```

Then train with `driver.train_dualdistil`, e.g. add to a CoveR script:

```bash
--query_prefix "search_query:[unused0][unused1][unused2][unused3][unused4]" \
--num_views 5 --view_pooling cluster --view_start_idx 5 \
--aggregation_strategy mean \
--view_orthogonalize_method pairwise --view_orthogonalize_lambda 0.001 \
```

and encode and search queries with the same prefix, `--num_views` and `--view_pooling`.

### Ideas for a next attempt

- Supervise the views directly: match each view to a sub-question (Hungarian or soft assignment
  over `view · sq` similarities) instead of relying on orthogonality alone.
- Use `max` aggregation in training too, so a document can be "covered" by any one view. This
  matches retrieval with `--aggregation_strategy max`.
- Make `cluster` deterministic at inference (argmax).
- Fix the `cumulative` hard-coded `5`, the `weighted` shape bug and the `rrf` rank bug before
  comparing modes.
