from pathlib import Path
from typing import Any, Union

from embeddings.data.dataset import Dataset


class HuggingFaceDataset(Dataset[str]):
    def __init__(self, dataset: Union[str, Path], **load_dataset_kwargs: Any):
        super().__init__()
        self.dataset = dataset
        self.load_dataset_kwargs = load_dataset_kwargs
