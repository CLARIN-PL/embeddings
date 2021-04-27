import os
import pprint
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import srsly
from datasets import load_dataset, ClassLabel, DatasetDict, Dataset
from flair.data import Corpus

from flair.datasets import CSVClassificationDataset
from typing import Optional, Dict, List

from experimental.data.io import T_path
from experimental.datasets.base import BaseDataset
from experimental.datasets.utils.misc import all_files_exists
from experimental.defaults import DATASET_PATH
from experimental.utils.loggers import get_logger


class HuggingFaceClassificationDataset(BaseDataset):
    HUGGING_FACE_SUBSETS = ["train", "validation", "test"]

    def __init__(
        self,
        dataset_name: str,
        input_text_column_name: str,
        target_column_name: str,
        output_path: Optional[T_path] = None,
    ):
        self._logger = get_logger(str(self))

        self.output_path: Path = (
            Path(output_path) if output_path else DATASET_PATH.joinpath(dataset_name)
        )

        self.dataset_name = dataset_name
        self.input_text_column_name = input_text_column_name
        self.target_column_name = target_column_name

        if not all_files_exists(
            path=self.output_path, files=[x + ".csv" for x in self.HUGGING_FACE_SUBSETS]
        ):
            self.output_path.mkdir(parents=True, exist_ok=True)
            self.flair_datasets = self._preprocess()
        else:
            self.flair_datasets = self._load()

    def _preprocess(self) -> Dict[str, CSVClassificationDataset]:
        flair_datasets = {}

        hf_dataset = load_dataset(self.dataset_name)
        self._log_info(hf_dataset)

        for subset_name in hf_dataset.keys():
            self._check_compatibility(hf_dataset[subset_name])

        for subset_name in self.HUGGING_FACE_SUBSETS:
            if subset_name in hf_dataset.keys():
                flair_datasets[subset_name] = self._preprocess_subset(hf_dataset, subset_name)
            else:
                flair_datasets[subset_name] = None
        return flair_datasets

    def _log_info(self, hf_dataset: DatasetDict) -> None:
        self._logger.info(f"Dataset name: {self.dataset_name}")

        subsets_info = {
            subset: pprint.pformat(hf_dataset[subset].info.__dict__) for subset in hf_dataset.keys()
        }
        for k, v in groupby(subsets_info.items(), itemgetter(1)):
            self._logger.info(f"Info of {list(map(itemgetter(0), v))}:\n{k}")
        self._logger.info(f"Schemas:\t{hf_dataset}")

    def _preprocess_subset(
        self, hf_dataset: DatasetDict, subset_name: str
    ) -> CSVClassificationDataset:
        label_map = hf_dataset[subset_name].features[self.target_column_name].names
        hf_dataset[subset_name] = hf_dataset[subset_name].map(
            lambda row: {"named_target": label_map[row[self.target_column_name]]},
            remove_columns=[self.target_column_name],
        )

        hf_dataset[subset_name].to_csv(
            os.path.join(self.output_path, f"{subset_name}.csv"), header=False, index=False
        )

        column_name_map = {
            hf_dataset[subset_name].column_names.index(self.input_text_column_name): "text",
            hf_dataset[subset_name].column_names.index("named_target"): "label",
        }

        srsly.write_json(
            self.output_path.joinpath(f"{subset_name}_column_name_map.json"), column_name_map
        )

        return CSVClassificationDataset(
            os.path.join(self.output_path, f"{subset_name}.csv"), column_name_map
        )

    @staticmethod
    def _check_column_in_dataset(dataset: Dataset, column_name: str) -> None:
        if column_name not in dataset.features:
            raise KeyError(f"Column '{column_name}' not found in features.")

    def _check_classification_task(self, dataset: Dataset) -> None:
        if not isinstance(dataset.features[self.target_column_name], ClassLabel):
            raise ValueError(f"Type of target column is not '{ClassLabel.__name__}'.")

    def _check_compatibility(self, dataset: Dataset) -> None:
        self._check_column_in_dataset(dataset, self.input_text_column_name)
        self._check_column_in_dataset(dataset, self.target_column_name)
        self._check_classification_task(dataset)

    def _load(self) -> Dict[str, CSVClassificationDataset]:
        flair_datasets = {}
        for subset_name in self.HUGGING_FACE_SUBSETS:
            column_name_map = srsly.read_json(
                self.output_path.joinpath(f"{subset_name}_column_name_map.json")
            )
            column_name_map = {int(k): v for k, v in column_name_map.items()}
            flair_datasets[subset_name] = CSVClassificationDataset(
                self.output_path.joinpath(f"{subset_name}.csv"), column_name_map
            )
        return flair_datasets

    def to_flair_column_corpus(self) -> Corpus:
        if not self.flair_datasets["train"]:
            raise ValueError(
                f"Hugging Face dataset {self.dataset_name} does not have " f"TRAIN subset."
            )

        return Corpus(
            train=self.flair_datasets["train"],
            dev=self.flair_datasets["validation"],
            test=self.flair_datasets["test"],
            name=self.dataset_name,
        )

    def __repr__(self) -> str:
        return type(self).__name__
