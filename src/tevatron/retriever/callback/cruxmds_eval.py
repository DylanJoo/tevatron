from datasets import load_dataset
import torch
import os
import numpy as np
from tevatron.retriever.searcher import FaissFlatSearcher
from crux.evaluation.rac_eval import rac_eval
from crux.tools import load_run_or_qrel, load_diversity_qrel, load_ratings
from pathlib import Path

home = Path.home()
root_dir = os.environ.get('CRUX_ROOT')

class Validator:

    def __init__(self, collator, batch_size):
        self.query_dataset = load_dataset('DylanJHJ/valid-crux-mds')
        self.document_dataset = load_dataset('DylanJHJ/crux-mds-corpus')
        self.batch_size = 512
        self.collator = collator
        self.eval_counter = 0
        self.eval_every = 1

    def run(self, model, device):

        all_results = {}
        for split in list(self.query_dataset.keys()):
            run = {ex['query_id']: {} for ex in self.query_dataset[split]}

            # Encode queries
            self.collator.encode_is_query = True
            features = [(ex['query_id'], ex['query_text'], None) for ex in self.query_dataset[split]]
            candidates = [ex['document_ids'] for ex in self.query_dataset[split]]
            idx2qid, batch = self.collator(features)

            with torch.no_grad():
                model_output: EncoderOutput = model(query=batch.to(device))
                query_embs = model_output.q_reps.cpu().detach().numpy()

            # Encode documents (batch wise)
            self.collator.encode_is_query = False
            corpus = {ex['id']: ex['contents'] for ex in self.document_dataset['train']}
            corpus.update({ex['id']: ex['contents'] for ex in self.document_dataset['test']})

            for idx, document_ids in enumerate(candidates):
                features = [(docid, corpus[docid], None) for docid in document_ids]
                _, batch = self.collator(features)
                with torch.no_grad():
                    model_output: EncoderOutput = model(passage=batch.to(device))
                    doc_embs = model_output.p_reps.cpu().detach().numpy()
                    scores = query_embs[idx] @ doc_embs.T # (N H) or (H) x (H D) = (N, D) or (D)
                    if scores.ndim == 2: # only check one qid
                        scores = scores.mean(axis=0)

                # assign scores
                qid = idx2qid[idx]
                run[qid] = { docid: float(s) for docid, s in zip(document_ids, scores) }

            # evaluation
            qrel = load_run_or_qrel(
                f'{root_dir}/crux-mds-{split}/qrels/div_qrels-tau3.txt', 
                threshold=1
            )
            div_qrel = load_diversity_qrel(
                f'{root_dir}/crux-mds-{split}/qrels/div_qrels-tau3.txt'
            )
            qrel = {k: v for k, v in qrel.items() if k in run}
            div_qrel = div_qrel[div_qrel['query_id'].isin(run.keys())]

            ratings = load_ratings(f'{root_dir}/crux-mds-{split}/judge')
            outputs = rac_eval(
                run=run,
                qrel=qrel, 
                div_qrel=div_qrel,
                tau=3,
                cutoff=10,
                judge=ratings,
                filter_by_oracle=True
            )
            for key, values in outputs.items():
                avg_value = np.mean(values)
                all_results[f"{split}-{key}"] = avg_value

        return all_results

