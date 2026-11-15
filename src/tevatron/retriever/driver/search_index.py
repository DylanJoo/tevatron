"""Search compressed faiss index shards built by `encode_and_index.py` (SQ8, SQ4, PQ96, ...).

Counterpart of `search.py`, which searches the float32 embeddings of `encode.py`. Queries are encoded with
`encode.py`. Shards are loaded and searched one at a time and merged into a running top-k, so memory stays at one
shard plus the top-k lists however large the corpus is. With exhaustive (non-IVF) indexes that share one trained
index, the result is the same as searching all shards as one index.

Output: the same format as `search.py` (pickle of (scores, passage ids), or text with --save_text).
"""
import glob
import logging
import pickle
from argparse import ArgumentParser

import faiss
import numpy as np
from tqdm import tqdm

from tevatron.retriever.driver.search import pickle_load

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def search_shards(index_files, q_reps, depth):
    """Search each shard and merge into a running top-`depth` per query. Returns (scores, passage ids);
    slots left empty when the corpus has fewer than `depth` passages have score -inf and id ''."""
    scores = np.full((len(q_reps), depth), -np.inf, dtype=np.float32)
    ids = np.full((len(q_reps), depth), '', dtype=object)
    for path in tqdm(index_files, desc='Searching shards'):
        with open(path, 'rb') as f:
            index_bytes, lookup = pickle.load(f)
        index = faiss.deserialize_index(index_bytes)
        del index_bytes
        s, i = index.search(q_reps, depth)
        lookup = np.asarray(lookup, dtype=object)
        s_ids = np.where(i >= 0, lookup[np.maximum(i, 0)], '')
        s = np.where(i >= 0, s, -np.inf)
        all_s, all_ids = np.concatenate([scores, s], axis=1), np.concatenate([ids, s_ids], axis=1)
        top = np.argsort(-all_s, axis=1, kind='stable')[:, :depth]
        scores, ids = np.take_along_axis(all_s, top, axis=1), np.take_along_axis(all_ids, top, axis=1)
        del index, lookup
    return scores, ids


def main():
    parser = ArgumentParser()
    parser.add_argument('--query_reps', required=True, help='query embedding pickle from encode.py')
    parser.add_argument('--passage_index', required=True, help='glob of shard pickles from encode_and_index.py')
    parser.add_argument('--depth', type=int, default=1000)
    parser.add_argument('--save_ranking_to', required=True)
    parser.add_argument('--save_text', action='store_true')
    args = parser.parse_args()

    index_files = sorted(glob.glob(args.passage_index))
    if not index_files:
        raise FileNotFoundError(f'No index files match {args.passage_index}')
    logger.info(f'Pattern match found {len(index_files)} index files.')

    q_reps, q_lookup = pickle_load(args.query_reps)
    scores, ids = search_shards(index_files, np.ascontiguousarray(q_reps, dtype=np.float32), args.depth)

    if args.save_text:
        with open(args.save_ranking_to, 'w') as f:  # rows are sorted by score; empty slots are skipped
            for qid, q_scores, q_ids in zip(q_lookup, scores, ids):
                f.writelines(f'{qid}\t{pid}\t{s}\n' for s, pid in zip(q_scores, q_ids) if pid != '')
    else:
        with open(args.save_ranking_to, 'wb') as f:
            pickle.dump((scores, ids), f)


if __name__ == '__main__':
    main()
