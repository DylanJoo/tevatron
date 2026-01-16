import faiss
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import logging

logger = logging.getLogger(__name__)


class FaissFlatSearcher:
    def __init__(self, init_reps: np.ndarray):
        index = faiss.IndexFlatIP(init_reps.shape[1])
        self.index = index

    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)

    def search(self, q_reps: np.ndarray, k: int):
        return self.index.search(q_reps, k)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int, quiet: bool=False, aggregation_strategy: str='sum'):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size), disable=quiet):
            if q_reps.ndim == 2:
                nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
                all_scores.append(nn_scores)
                all_indices.append(nn_indices)
            else:
                nn_scores, nn_indices = self.parallel_search(q_reps[start_idx: start_idx + batch_size], k, aggregation_strategy)
                all_scores.append(nn_scores)
                all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices

    def parallel_search(self, sq_reps: np.ndarray, k: int, aggregation_strategy: str):
        """ The q_reps should be a batch of subqueries.  """
        batch_size = sq_reps.shape[0]
        num_subqueries = sq_reps.shape[1]
        sq_reps_flatten = sq_reps.reshape(-1, sq_reps.shape[-1])
        nn_scores, nn_indices = self.search(sq_reps_flatten, k)
        num_docs = 0

        batch_indices, batch_scores = [], []
        for batch_idx in range(batch_size):
            start = batch_idx * num_subqueries
            end   = (batch_idx + 1) * num_subqueries
            batch_nn_scores  = nn_scores[start: end]
            batch_nn_indices = nn_indices[start: end]

            # score-sum or score-max
            doc2score = defaultdict(float)
            for score, docid in zip(batch_nn_scores.flatten(), batch_nn_indices.flatten()):
                if aggregation_strategy == 'max':
                    doc2score[docid] = max(score, doc2score[docid])
                else:
                    doc2score[docid] += score

            # sorted and return the top-k
            sorted_docs = sorted(doc2score.items(), key=lambda x: x[1], reverse=True)
            top_docs = sorted_docs[:k] # NOTE: maybe returning more than k is fine (but need to handle numpy matrix)
            num_docs += len(sorted_docs)

            batch_scores.append(np.array([s for _, s in top_docs]))
            batch_indices.append(np.array([d for d, _ in top_docs]))

        print('Average # retrieved documents:', num_docs / batch_size)
        return batch_scores, batch_indices


class FaissSearcher(FaissFlatSearcher):

    def __init__(self, init_reps: np.ndarray, factory_str: str):
        index = faiss.index_factory(init_reps.shape[1], factory_str)
        self.index = index
        self.index.verbose = True
        if not self.index.is_trained:
            self.index.train(init_reps)
