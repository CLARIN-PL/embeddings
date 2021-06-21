import abc
import pprint
import tempfile
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Dict

import datasets
from flair.data import Corpus, FlairDataset

from embeddings.defaults import DATASET_PATH
from embeddings.transformation.transformation import Transformation
from embeddings.utils.loggers import get_logger

_logger = get_logger(__name__)


class CorpusTransformation(Transformation[datasets.DatasetDict, Corpus]):
    HUGGING_FACE_SUBSETS = ["train", "validation", "test"]

    def __init__(
        self,
        input_column_name: str,
        target_column_name: str,
        datasets_path: Path = DATASET_PATH,
    ):
        super().__init__()

        self.datasets_cache_path = datasets_path
        self.input_column_name = input_column_name
        self.target_column_name = target_column_name

    def transform(self, data: datasets.DatasetDict) -> Corpus:
        with tempfile.TemporaryDirectory() as tmp_dir_path:
            output_path = Path(tmp_dir_path)
            flair_datasets = self._preprocess(data, output_path)
        return self._to_flair_corpus(flair_datasets)

    @staticmethod
    def _to_flair_corpus(flair_datasets: Dict[str, FlairDataset]) -> Corpus:
        if not flair_datasets["train"]:
            raise ValueError("Hugging Face dataset does not contain TRAIN subset.")

        return Corpus(
            train=flair_datasets["train"],
            dev=flair_datasets["validation"],
            test=flair_datasets["test"],
        )

    def _preprocess(
        self, hf_datadict: datasets.DatasetDict, output_path: Path
    ) -> Dict[str, FlairDataset]:
        flair_datasets = {}
        self._log_info(hf_datadict)

        for subset_name in hf_datadict.keys():
            self._check_compatibility(hf_datadict[subset_name])

        for subset_name in CorpusTransformation.HUGGING_FACE_SUBSETS:
            if subset_name in hf_datadict.keys():
                flair_datasets[subset_name] = self._preprocess_subset(
                    hf_datadict, subset_name, output_path
                )
            else:
                flair_datasets[subset_name] = None
        return flair_datasets

    @abc.abstractmethod
    def _preprocess_subset(
        self, hf_datadict: datasets.DatasetDict, subset_name: str, output_path: Path
    ) -> FlairDataset:
        pass

    @staticmethod
    def _log_info(hf_datadict: datasets.DatasetDict) -> None:
        subsets_info = {
            subset: pprint.pformat(hf_datadict[subset].info.__dict__)
            for subset in hf_datadict.keys()
        }
        for k, v in groupby(subsets_info.items(), itemgetter(1)):
            _logger.info(f"Info of {list(map(itemgetter(0), v))}:\n{k}")
        _logger.info(f"Schemas:\t{hf_datadict}")

    def _check_compatibility(self, dataset: datasets.Dataset) -> None:
        self._check_column_in_dataset(dataset, self.input_column_name)
        self._check_column_in_dataset(dataset, self.target_column_name)
        self._check_task(dataset)

    @staticmethod
    def _check_column_in_dataset(dataset: datasets.Dataset, column_name: str) -> None:
        if column_name not in dataset.features:
            raise KeyError(f"Column '{column_name}' not found in features.")

    @abc.abstractmethod
    def _check_task(self, dataset: datasets.Dataset) -> None:
        """Checking if dataset is compatible with the given task."""
        pass
