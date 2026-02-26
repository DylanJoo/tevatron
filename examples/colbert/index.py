import logging
import os
import sys
import pickle
from dataclasses import dataclass, field
from typing import Optional

from tqdm import tqdm
from transformers import HfArgumentParser

from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.dataset import EncodeDataset

logger = logging.getLogger(__name__)


@dataclass
class ColBERTArguments:
    nbits: int = field(
        default=2,
        metadata={"help": "Number of bits per dimension for ColBERT residual compression"}
    )
    step: str = field(
        default='prepare',
        metadata={"help": "Indexing step: prepare, encode, or finalize. If None, runs full pipeline."}
    )
    n_gpus: Optional[int] = field(
        default=None,
        metadata={"help": "Number of GPUs to use. Defaults to all available."}
    )
    index_name: Optional[str] = field(
        default=None,
        metadata={"help": "Override index name (default: basename of encode_output_path)"}
    )
    experiment: str = field(
        default="default",
        metadata={"help": "Name of the experiment (ColBERT Run context)"}
    )
    root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory for ColBERT Run context"}
    )
    base_model: Optional[str] = field(
        default=None,
        metadata={"help": "Base model for v1 checkpoints"}
    )
    max_num_partitions: int = field(
        default=-1,
        metadata={"help": "Max number of partitions/centroids used in indexing"}
    )
    max_sampled_pid: int = field(
        default=-1,
        metadata={"help": "Max number of sampled tokens for training KMeans"}
    )
    lazy_collection_loader: bool = field(
        default=False,
        metadata={"help": "Use lazy (offset-map) collection loading"}
    )
    ivf_num_processes: int = field(
        default=20,
        metadata={"help": "Number of processes for IVF building"}
    )
    ivf_use_tempdir: bool = field(
        default=False,
        metadata={"help": "Use a temp directory for IVF building"}
    )
    ivf_merging_ways: int = field(
        default=2,
        metadata={"help": "Merging ways for IVF"}
    )
    use_lagacy_build_ivf: bool = field(
        default=False,
        metadata={"help": "Use legacy IVF build method"}
    )
    reuse_centroids_from: Optional[str] = field(
        default=None,
        metadata={"help": "Reuse centroids from an existing index"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ColBERTArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, colbert_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, colbert_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments
        colbert_args: ColBERTArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    if data_args.encode_is_query:
        raise ValueError("index.py is for corpus indexing only. Set encode_is_query=False.")

    print("# Model arguments:", model_args)
    print("# Data arguments:", data_args)
    print("# ColBERT arguments:", colbert_args)

    # Load corpus via tevatron EncodeDataset
    encode_dataset = EncodeDataset(data_args=data_args)

    collection = []
    document_ids = []
    for item in tqdm(encode_dataset, desc="Loading corpus"):
        content_id, content_text = item[0], item[1]
        if content_text:
            collection.append(content_text)
            document_ids.append(str(content_id))

    logger.info(f"Loaded {len(collection)} documents for indexing.")

    from colbert.data import Collection
    collection = Collection(data=collection)

    # Derive index_root and index_name from encode_output_path
    index_root = os.path.dirname(os.path.abspath(data_args.encode_output_path))
    index_name = colbert_args.index_name or os.path.basename(data_args.encode_output_path)
    os.makedirs(index_root, exist_ok=True)

    # Save docid mapping (integer position -> original docid) for use during search
    docid_map_path = os.path.join(index_root, f"{index_name}.docid_map.pkl")
    with open(docid_map_path, "wb") as f:
        pickle.dump(document_ids, f)
    logger.info(f"Docid mapping saved to {docid_map_path}")

    # Determine GPU count
    if colbert_args.n_gpus is None:
        import torch
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = colbert_args.n_gpus

    # Build the ColBERT PLAID index
    from colbert import Indexer
    from colbert.infra import Run, RunConfig, ColBERTConfig

    config = ColBERTConfig(
        doc_maxlen=data_args.passage_max_len,
        nbits=colbert_args.nbits,
        max_sampled_pid=colbert_args.max_sampled_pid,
        max_num_partitions=colbert_args.max_num_partitions,
    )

    if colbert_args.base_model is not None:
        config.configure(model_name=colbert_args.base_model)
    if model_args.model_name_or_path.endswith('.dnn'):
        config.configure(force_resize_embeddings=True, mask_punctuation=False)

    config.configure(
        index_name=index_name,
        use_lagacy_build_ivf=colbert_args.use_lagacy_build_ivf,
        reuse_centroids_from=colbert_args.reuse_centroids_from,
    )

    run_config = RunConfig(
        nranks=n_gpus,
        root=colbert_args.root,
        ivf_num_processes=colbert_args.ivf_num_processes,
        ivf_use_tempdir=colbert_args.ivf_use_tempdir,
        ivf_merging_ways=colbert_args.ivf_merging_ways,
        experiment=colbert_args.experiment,
        index_root=index_root,
    )

    with Run().context(run_config):
        indexer = Indexer(checkpoint=model_args.model_name_or_path, config=config)

        if colbert_args.step == 'prepare':
            indexer.prepare(name=index_name, collection=collection, overwrite='resume')
        elif colbert_args.step == 'prepare-gpu':
            indexer.prepare(name=index_name, collection=collection, overwrite='resume', no_kmeans=True)
        elif colbert_args.step == 'prepare-cpu':
            indexer.prepare(name=index_name, collection=collection, overwrite='resume', no_sample=True)
        elif colbert_args.step == 'encode':
            indexer.encode(name=index_name, collection=collection)
        elif colbert_args.step == 'finalize':
            indexer.finalize(name=index_name, collection=collection)
            print(indexer.get_index())
            logger.info("Index created.")
        else:
            # Full pipeline
            indexer.index(name=index_name, collection=collection, overwrite=True)
            logger.info(f"Index saved to {data_args.encode_output_path}")


if __name__ == "__main__":
    main()
