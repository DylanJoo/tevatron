import pickle

import numpy as np
import glob
from argparse import ArgumentParser
from itertools import chain
from tqdm import tqdm
import faiss

from tevatron.retriever.searcher import FaissFlatSearcher

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def search_queries(retriever, q_reps, p_lookup, args):
    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size, args.quiet)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth)

    psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices

def search_mmr_queries(retriever, 
                       q_reps, 
                       q_lookup,
                       p_reps, 
                       p_lookup, 
                       args):

    p_lookup_r = {pid: idx for idx, pid in enumerate(p_lookup)}
    all_scores, psg_indices = search_queries(retriever, q_reps, p_lookup, args)

    mmr_indices = []
    mmr_scores = []

    ## iterate over queries
    for i, qid in enumerate(q_lookup):
        
        ## Gather document embeddings and scores
        pids = psg_indices[i].tolist()
        pscores = all_scores[i]
        pembs = np.array([p_reps[p_lookup_r[pid]] for pid in pids])
        assert len(pids) == len(pscores), 'Got passages length: {len(pids)} and {len(pscores)}'
        sim_matrix = pembs @ pembs.T
        
        ## MMR selection
        selected_indices = []
        selected_scores = []
        remaining_indices = set(range(len(pids)))
        
        for _ in range(args.depth):
            best_idx = None
            best_mmr = float('-inf')
            
            for idx in remaining_indices:
                pscore = pscores[idx]
                
                # Compute max similarity to selected documents
                if len(selected_indices) == 0:
                    max_sim = 0.0
                else:
                    sims_to_selected = sim_matrix[idx, selected_indices]
                    max_sim = np.max(sims_to_selected)
                
                # MMR formula
                mmr_score = args.lambda_param * pscore - (1 - args.lambda_param) * max_sim
                # mmr_score = pscore
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                selected_scores.append(best_mmr)
                remaining_indices.remove(best_idx)

        # Concatenate the ranking of rest of them? after top-k
        selected_indices.extend([idx for idx in range(len(pids)) if idx in remaining_indices])
        selected_psg_indices = [pids[idx] for idx in selected_indices]
        selected_scores.extend([pscore for pid, pscore in zip(pids, pscores) if pid in remaining_indices])

        ## Convert to document id
        selected_indices = [pids[idx] for idx in selected_indices]

        ## Reconstructed mmr results
        mmr_indices.append(selected_indices)
        mmr_scores.append(selected_scores)

    return mmr_scores, mmr_indices
        
def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file):
    with open(ranking_save_file, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for s, idx in score_list:
                f.write(f'{qid}\t{idx}\t{s}\n')


def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def main():
    parser = ArgumentParser()
    parser.add_argument('--query_reps', required=True)
    parser.add_argument('--passage_reps', required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--depth', type=int, default=1000)
    parser.add_argument('--save_ranking_to', required=True)
    parser.add_argument('--save_text', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--lambda_param', type=float, default=0.9)

    args = parser.parse_args()

    index_files = glob.glob(args.passage_reps)
    logger.info(f'Pattern match found {len(index_files)} files; loading them into index.')

    p_reps_0, p_lookup_0 = pickle_load(index_files[0])
    retriever = FaissFlatSearcher(p_reps_0)

    shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
    if len(index_files) > 1:
        shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))

    look_up = []
    all_p_reps = []
    for p_reps, p_lookup in shards:
        retriever.add(p_reps)
        look_up += p_lookup
        all_p_reps.append(p_reps)

    all_p_reps = np.vstack(all_p_reps)
    q_reps, q_lookup = pickle_load(args.query_reps)
    q_reps = q_reps

    num_gpus = faiss.get_num_gpus()
    if num_gpus == 0:
        logger.info("No GPU found or using faiss-cpu. Back to CPU.")
    else:
        logger.info(f"Using {num_gpus} GPU")
        if num_gpus == 1:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            res = faiss.StandardGpuResources()
            retriever.index = faiss.index_cpu_to_gpu(res, 0, retriever.index, co)
        else:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            retriever.index = faiss.index_cpu_to_all_gpus(retriever.index, co,
                                                     ngpu=num_gpus)

    logger.info('Index Search Start')
    all_scores, psg_indices = search_mmr_queries(retriever, 
                                                 q_reps, 
                                                 q_lookup,
                                                 all_p_reps, 
                                                 look_up, 
                                                 args)
    logger.info('Index Search Finished')

    if args.save_text:
        write_ranking(psg_indices, all_scores, q_lookup, args.save_ranking_to)
    else:
        pickle_save((all_scores, psg_indices), args.save_ranking_to)


if __name__ == '__main__':
    main()
