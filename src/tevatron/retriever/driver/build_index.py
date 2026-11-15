"""Build faiss index shards (Flat, SQ8, PQ64, ...) from the saved embeddings of `encode.py` -- CPU only, no model.

One output shard per input embedding pickle, in the format of `encode_and_index.py`, so `search_index.py` searches
it. `--index_factory Flat` gives the exact (uncompressed) baseline through the same search path.

All shards share one trained index. Training uses the same number of randomly sampled vectors (claims) from every
input file, so every file/collection is represented:
  1. train once:        --passage_reps '<all pkls>' --trained_index_path T            (no --file_index)
  2. add, in parallel:  --passage_reps '<all pkls>' --trained_index_path T --file_index $i
Without --file_index and with --output_dir, step 1 continues and builds all shards in this process.
Flat needs no training: skip step 1 and leave out --trained_index_path.
"""
import glob
import logging
import os
import pickle
from argparse import ArgumentParser

import faiss
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def load_reps(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.ascontiguousarray(reps, dtype=np.float32), lookup  # faiss needs float32 (pickles may be float16)


def sample_training_vectors(files, train_size, seed):
    """`train_size / #files` random vectors from each file."""
    rng = np.random.default_rng(seed)
    per_file = -(-train_size // len(files))  # ceil
    sample = []
    for path in tqdm(files, desc=f'Sampling {per_file} vectors per file'):
        reps, _ = load_reps(path)
        sample.append(reps[np.sort(rng.choice(len(reps), size=min(per_file, len(reps)), replace=False))])
    return np.concatenate(sample)


def main():
    parser = ArgumentParser()
    parser.add_argument('--passage_reps', required=True, help='glob of ALL embedding pickles from encode.py')
    parser.add_argument('--index_factory', default='SQ8', help="faiss index_factory string: Flat, SQ8, SQ4, PQ64, ...")
    parser.add_argument('--trained_index_path', default=None, help='trained, empty index shared by all shards')
    parser.add_argument('--index_train_size', type=int, default=200_000, help='vectors sampled (over all files) to train')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default=None, help='where the index shards go (<input name>.<factory>.pkl)')
    parser.add_argument('--file_index', type=int, default=None, help='only build the shard of the i-th (sorted) input file')
    args = parser.parse_args()

    files = sorted(glob.glob(args.passage_reps))
    if not files:
        raise FileNotFoundError(f'No files match {args.passage_reps}')
    logger.info(f'{len(files)} embedding files')

    # 1. Trained, empty index
    if args.trained_index_path and os.path.exists(args.trained_index_path):
        logger.info(f'Loading trained index from {args.trained_index_path}')
        index = faiss.read_index(args.trained_index_path)
    else:
        if args.file_index is not None and args.trained_index_path:
            raise FileNotFoundError(f'{args.trained_index_path} does not exist: run the train step first (no --file_index).')
        dim = load_reps(files[0])[0].shape[1]
        index = faiss.index_factory(dim, args.index_factory, faiss.METRIC_INNER_PRODUCT)
        if not index.is_trained:
            train = sample_training_vectors(files, args.index_train_size, args.seed)
            logger.info(f'Training "{args.index_factory}" on {len(train)} vectors of dim {dim}')
            index.train(train)
            del train
        if args.trained_index_path:
            os.makedirs(os.path.dirname(os.path.abspath(args.trained_index_path)), exist_ok=True)
            faiss.write_index(index, args.trained_index_path)
            logger.info(f'Saved trained index to {args.trained_index_path}')

    if args.output_dir is None:
        logger.info('No --output_dir: trained the index only.')
        return

    # 2. One shard per input file
    os.makedirs(args.output_dir, exist_ok=True)
    todo = [files[args.file_index]] if args.file_index is not None else files
    factory_name = args.index_factory.replace(',', '_')
    for path in tqdm(todo, desc='Building shards', disable=len(todo) == 1):
        out = os.path.join(args.output_dir, f'{os.path.splitext(os.path.basename(path))[0]}.{factory_name}.pkl')
        reps, lookup = load_reps(path)
        shard = faiss.clone_index(index)
        shard.add(reps)
        del reps
        with open(out, 'wb') as f:
            pickle.dump((faiss.serialize_index(shard), lookup), f, protocol=4)
        logger.info(f'Saved {shard.ntotal} vectors to {out}')


if __name__ == '__main__':
    main()
