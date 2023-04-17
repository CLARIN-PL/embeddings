from typing import Any

import datasets

from embeddings.data.io import T_path
from embeddings.utils.results_persister import ResultsPersister


class HuggingFaceDatasetLocalPersister(ResultsPersister[datasets.DatasetDict]):
    def __init__(self, path: T_path) -> None:
        self.path = path

    def persist(self, result: datasets.DatasetDict, **kwargs: Any) -> None:
        result.save_to_disk(self.path)
