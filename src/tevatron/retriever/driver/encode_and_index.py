import logging
import os
import pickle
import glob
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import faiss
from tqdm import tqdm
import torch

from torch.utils.data import ConcatDataset, DataLoader, Subset
from transformers import AutoTokenizer, HfArgumentParser

from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.dataset import EncodeDataset
from tevatron.retriever.collator import EncodeCollator
from tevatron.retriever.modeling import EncoderOutput, DenseModel

logger = logging.getLogger(__name__)


@dataclass
class IndexArguments:
    index_factory: str = field(default="SQ8", metadata={"help": "faiss index_factory string: SQ4 (default), SQ8, PQ96, ..."})
    trained_index_path: Optional[str] = field(
        default=None, metadata={"help": "trained, empty index shared by all shards: loaded if it exists, "
                                        "otherwise trained and written here"})
    index_output_path: Optional[str] = field(
        default=None, metadata={"help": "output pickle of this shard; omit to only train the index"})
    index_train_size: int = field(default=100_000, metadata={"help": "passages sampled to train the index"})
    index_train_files: Optional[str] = field(
        default=None, metadata={"help": "train-only mode: glob of ALL corpus files; index_train_size / #files "
                                        "passages are sampled from each file. Default: sample from --dataset_path"})

def encode(model, dataset, collator, training_args):
    """The encoding loop of encode.py (passages only): returns (float32 embeddings [n, dim], ids)."""
    loader = DataLoader(
        dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    encoded = []
    lookup_indices = []
    for (batch_ids, batch) in tqdm(loader):
        lookup_indices.extend(batch_ids)
        with torch.amp.autocast('cuda') if training_args.fp16 or training_args.bf16 else nullcontext():
            with torch.no_grad():
                batch = {k: v.to(training_args.device) for k, v in batch.items()}
                model_output: EncoderOutput = model(passage=batch)
                encoded.append(model_output.p_reps.float().cpu().detach().numpy())  # faiss needs float32
    return np.ascontiguousarray(np.concatenate(encoded)), lookup_indices


def sample_train_set(data_args, index_args):
    """Uniform sample of the dataset, or -- with --index_train_files -- the same number of passages from each file."""
    rng = np.random.default_rng(42)
    if index_args.index_train_files is None:
        datasets = [EncodeDataset(data_args=data_args)]
    else:
        files = sorted(glob.glob(index_args.index_train_files))
        if not files:
            raise FileNotFoundError(f'No files match {index_args.index_train_files}')
        datasets = [EncodeDataset(data_args=replace(data_args, dataset_path=f, dataset_number_of_shards=1, dataset_shard_index=0))
                    for f in files]
    per_dataset = -(-index_args.index_train_size // len(datasets))  # ceil
    subsets = []
    for dataset in datasets:
        sample = np.sort(rng.choice(len(dataset), size=min(per_dataset, len(dataset)), replace=False))
        subsets.append(Subset(dataset, sample.tolist()))
    logger.info(f'Training sample: {per_dataset} passages from each of {len(datasets)} file(s)')
    return ConcatDataset(subsets)


def train_index(model, train_set, collator, index_args, training_args):
    """Train an empty index on the embeddings of the sampled passages."""
    reps, _ = encode(model, train_set, collator, training_args)
    logger.info(f'Training "{index_args.index_factory}" on {len(reps)} vectors of dim {reps.shape[1]}')
    index = faiss.index_factory(reps.shape[1], index_args.index_factory, faiss.METRIC_INNER_PRODUCT)
    index.train(reps)
    return index


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, IndexArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, index_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, index_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    index_args: IndexArguments
    training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')
    if data_args.encode_is_query:
        raise ValueError('encode_and_index.py builds a passage index; encode queries with encode.py.')
    if index_args.index_output_path is None and index_args.trained_index_path is None:
        raise ValueError('Set --index_output_path, or only --trained_index_path to train the index.')
    if data_args.dataset_number_of_shards > 1 and index_args.trained_index_path is None:
        raise ValueError('Sharded encoding needs --trained_index_path so that all shards share one trained index.')
    if index_args.index_train_files and index_args.index_output_path:
        raise ValueError('--index_train_files is for the train-only job; drop --index_output_path there.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    print("# The model arguemnts:", model_args)
    print("# The data arguemnts:", data_args)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.padding_side == 'right':
        tokenizer.padding_side = 'right'
    else:
        tokenizer.padding_side = 'left'

    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = DenseModel.load(
        model_args.model_name_or_path,
        pooling=model_args.pooling,
        normalize=model_args.normalize,
        lora_name_or_path=model_args.lora_name_or_path,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )

    if isinstance(data_args.dataset_name, list):
        data_args.dataset_name = data_args.dataset_name[0]
        data_args.dataset_split = data_args.dataset_split[0]

    # not needed by the train-only job, which reads --index_train_files instead
    encode_dataset = EncodeDataset(data_args=data_args) if index_args.index_output_path else None

    encode_collator = EncodeCollator(
        data_args=data_args,
        tokenizer=tokenizer,
    )

    model = model.to(training_args.device)
    model.eval()

    # 1. Get a trained, empty index
    trained_path = index_args.trained_index_path
    train_only = index_args.index_output_path is None

    ## indexing a shard: load the index trained once, by the train-only job, for all shards
    if trained_path and not train_only:
        if not os.path.exists(trained_path):
            raise FileNotFoundError(f'{trained_path} does not exist: run the train-only job first '
                                    '(--trained_index_path + --index_train_files, no --index_output_path).')
        logger.info(f'Loading trained index from {trained_path}')
        index = faiss.read_index(trained_path)

    ## train-only job (sample from every file), or a single unsharded corpus (sample from it)
    else:
        train_set = sample_train_set(data_args, index_args)
        index = train_index(model, train_set, encode_collator, index_args, training_args)
        if trained_path:
            os.makedirs(os.path.dirname(os.path.abspath(trained_path)), exist_ok=True)
            faiss.write_index(index, trained_path)
            logger.info(f'Saved trained index to {trained_path}')
    assert index.is_trained and index.ntotal == 0, 'expected a trained, empty index'

    if index_args.index_output_path is None:
        logger.info('No --index_output_path: trained the index only.')
        return

    # 2. Encode the shard (as encode.py) and add it to the index
    encoded, lookup_indices = encode(model, encode_dataset, encode_collator, training_args)
    index.add(encoded)
    assert index.ntotal == len(lookup_indices)

    # 3. Save (compressed index, ids) instead of (embeddings, ids)
    os.makedirs(os.path.dirname(os.path.abspath(index_args.index_output_path)), exist_ok=True)
    with open(index_args.index_output_path, 'wb') as f:
        pickle.dump((faiss.serialize_index(index), lookup_indices), f, protocol=4)
    logger.info(f'Saved {index.ntotal} vectors ({index_args.index_factory}) to {index_args.index_output_path}')


if __name__ == "__main__":
    main()
