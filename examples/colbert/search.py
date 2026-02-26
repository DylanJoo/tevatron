import logging
import os
import sys
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from tqdm import tqdm
from transformers import HfArgumentParser

from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.dataset import EncodeDataset

logger = logging.getLogger(__name__)


@dataclass
class SearchArguments:
    index_path: str = field(
        metadata={"help": "Path to the ColBERT index (encode_output_path used during indexing)"}
    )
    run_path: str = field(metadata={"help": "Path to the result"})
    depth: int = field(
        default=100,
        metadata={"help": "Number of results to retrieve per query"}
    )
    nbits: int = field(
        default=2,
        metadata={"help": "Number of bits for ColBERT quantization (must match the index)"}
    )
    only_approx: bool = field(
        default=False,
        metadata={"help": "Use approximate search (faster but less accurate)"}
    )


def maxp(passage_triples: List[Tuple[int, int, float]], mapping) -> Dict[str, float]:
    doc_score = {}
    for pid, _, score in passage_triples:
        doc_id = mapping[pid]
        if doc_id not in doc_score or doc_score[doc_id] < score:
            doc_score[doc_id] = score
    return doc_score


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, SearchArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, search_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, search_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments
        search_args: SearchArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    if not data_args.encode_is_query:
        raise ValueError("search.py is for query retrieval. Set encode_is_query=True.")

    # Load docid mapping saved during indexing: list of original docids indexed by integer position
    index_root = os.path.dirname(os.path.abspath(search_args.index_path))
    index_name = os.path.basename(search_args.index_path)
    docid_map_path = os.path.join(index_root, f"{index_name}.docid_map.pkl")

    with open(docid_map_path, "rb") as f:
        docid_map = pickle.load(f)
    logger.info(f"Loaded docid mapping ({len(docid_map)} entries) from {docid_map_path}")

    # Load queries via tevatron EncodeDataset
    encode_dataset = EncodeDataset(data_args=data_args)

    queries = {}  # qid -> query_text, preserving insertion order for tqdm
    for item in tqdm(encode_dataset, desc="Loading queries"):
        content_id, content_text = item[0], item[1]
        if content_text:
            queries[str(content_id)] = content_text

    logger.info(f"Loaded {len(queries)} queries.")

    # Search with colbert-ai
    from colbert import Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig

    save_dir = os.path.dirname(os.path.abspath(search_args.run_path))
    os.makedirs(save_dir, exist_ok=True)

    with Run().context(RunConfig(nranks=1, index_root=index_root)):
        config = ColBERTConfig(
            query_maxlen=data_args.query_max_len,
            nbits=search_args.nbits,
        )
        searcher = Searcher(index=index_name, config=config)
        searcher.config.configure(only_approx=search_args.only_approx, ignore_unrecognized=False)

        rankings = searcher.search_all(queries, k=search_args.depth)

    with open(search_args.run_path, "w") as fout:
        for qid, passage_triples in tqdm(rankings.items(), desc="Writing results"):
            doc_scores = maxp(passage_triples, docid_map)
            for docid, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
                fout.write(f"{qid}\t{docid}\t{score}\n")

    logger.info(f"Rankings saved to {search_args.run_path}")


if __name__ == "__main__":
    main()
