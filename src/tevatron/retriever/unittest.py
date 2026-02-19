from tevatron.retriever.arguments import DataArguments
from tevatron.retriever.dataset_mixed import TrainDataset

data_args = DataArguments(
    dataseta_name = ['Tevatron/msmarco-passage', 'DylanJHJ/crux-researchy'],
    corpus_name
)
ds = TrainDataset(data_args)
