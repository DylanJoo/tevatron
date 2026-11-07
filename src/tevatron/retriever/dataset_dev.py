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
