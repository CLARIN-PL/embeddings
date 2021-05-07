from pathlib import Path
from typing import Union, Any

from embeddings.data.dataset import Data, Dataset


class HuggingFaceDataset(Dataset[Data]):
    def __init__(self, dataset: Union[str, Path], **load_dataset_kwargs: Any):
        super().__init__()
        self.dataset = dataset
        self.load_dataset_kwargs = load_dataset_kwargs
