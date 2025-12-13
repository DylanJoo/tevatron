import ir_measures
from ir_measures import nDCG 
import ir_datasets
from datasets import load_dataset
import torch
import numpy as np
from tevatron.retriever.searcher import FaissFlatSearcher

class Validator:

    def __init__(self, collator, batch_size):
        self.query_dataset = load_dataset('DylanJHJ/valid-nano-beir')
        self.document_dataset = load_dataset('DylanJHJ/beir-subset-corpus')
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
            split_ = split.replace('nano_', '')
            corpus = {ex['docid']: ex['text'] for ex in self.document_dataset[split_]}

            for idx, document_ids in enumerate(candidates):
                features = [(docid, corpus[docid], None) for docid in document_ids]
                _, batch = self.collator(features)
                with torch.no_grad():
                    model_output: EncoderOutput = model(passage=batch.to(device))
                    doc_embs = model_output.p_reps.cpu().detach().numpy()
                    scores = query_embs[idx] @ doc_embs.T

                # assign scores
                qid = idx2qid[idx]
                run[qid] = { docid: float(s) for docid, s in zip(document_ids, scores) }

            # evaluation
            irds_tag = split.replace("_", "-").replace(".", "/")
            qrels = ir_datasets.load(irds_tag).qrels_dict()
            all_results[split] = ir_measures.calc_aggregate([nDCG@10], qrels=qrels, run=run)[nDCG@10]
            # print({f"eval/{split}": all_results[split]})
        return all_results

