import random
import os
from typing import List, Tuple

from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from PIL import Image

from tevatron.retriever.arguments import DataArguments
from tevatron.retriever.dataset import TrainDataset, DistilTrainDataset, WideDistilTrainDataset

import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class CovDistilTrainDataset(TrainDataset):
    """
    Dataset for training which handles both query and passage data.
    Loads dataset and optional corpus from the provided paths/configurations.
    """

    def __getitem__(self, item):
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch) if self.trainer else 0
        _hashed_seed = hash(item + self.trainer.args.seed) if self.trainer else 0

        # Handling the new format
        query_id = group['query_id']
        query_text = group.get('query_text', '') or ''
        query_image = group.get('query_image', None)
        query_video = group.get('query_video', None)
        query_audio = group.get('query_audio', None)
        formatted_query = (self.data_args.query_prefix + query_text + self.data_args.query_postfix,
                           query_image, query_video, query_audio)

        formatted_documents = []
        positive_document_ids = group['positive_document_ids']
        negative_document_ids = group['negative_document_ids']
        subqueries = group['subquestions']

        # Select positive document id
        selected_positive_docid = positive_document_ids[(_hashed_seed + epoch) % len(positive_document_ids)]
        formatted_documents.append(
            self._get_info_from_docid(selected_positive_docid, self.data_args.passage_prefix)
        )

        # Select negative document ids
        negative_size = self.data_args.train_group_size - 1
        if len(negative_document_ids) < negative_size:
            selected_negative_docids = random.choices(negative_document_ids, k=negative_size)
        elif self.data_args.train_group_size == 1:
            selected_negative_docids = []
        else:
            offset = epoch * negative_size % len(negative_document_ids)
            selected_negative_docids = list(negative_document_ids)
            random.Random(_hashed_seed).shuffle(selected_negative_docids)
            selected_negative_docids = selected_negative_docids * 2
            selected_negative_docids = selected_negative_docids[offset: offset + negative_size]

        for neg_docid in selected_negative_docids:
            formatted_documents.append(
                self._get_info_from_docid(neg_docid, self.data_args.passage_prefix)
            )

        # Select subquery and mapping
        flatten_formatted_subqueries = [self.data_args.subquery_prefix + sq for sq in subqueries]

        return formatted_query, formatted_documents, flatten_formatted_subqueries

# class DualDistilTrainDataset(DistilTrainDataset):
class DualDistilTrainDataset(WideDistilTrainDataset):

    def __getitem__(self, item):
        ## inherit the DistilTrainDataset getitem
        formatted_query, formatted_documents, formatted_scores = super().__getitem__(item) # the distillation

        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch) if self.trainer else 0
        _hashed_seed = hash(item + self.trainer.args.seed) if self.trainer else 0

        # Handling the new format
        subqueries = group['subquestions']
        flatten_formatted_subqueries = [self.data_args.subquery_prefix + sq for sq in subqueries]

        return formatted_query, formatted_documents, flatten_formatted_subqueries, formatted_scores
