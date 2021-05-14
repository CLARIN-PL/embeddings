import datasets

from embeddings.data.data_loader import DataLoader, Input
from embeddings.data.dataset import Dataset
from embeddings.data.hugging_face_dataset import HuggingFaceDataset


class HuggingFaceDataLoader(DataLoader[Input, datasets.DatasetDict]):
    def load(self, dataset: Dataset[Input]) -> datasets.DatasetDict:
        if isinstance(dataset, HuggingFaceDataset):
            return datasets.load_dataset(dataset.dataset, **dataset.load_dataset_kwargs)
        else:
            raise ValueError('This DataLoader should be used with HuggingFaceDataset only.')
