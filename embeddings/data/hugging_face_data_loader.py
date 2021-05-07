import os
import pprint
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Dict

import datasets
import srsly
from flair.data import Corpus
from flair.datasets import CSVClassificationDataset

from embeddings.data.dataset import Dataset, Data
from embeddings.data.hugging_face_dataset import HuggingFaceDataset
from experimental.datasets.utils.misc import all_files_exists
from experimental.defaults import DATASET_PATH


class HuggingFaceClassificationDataLoader(Dataset[Data]):
    HUGGING_FACE_SUBSETS = ["train", "validation", "test"]

    def __init__(
        self,
        input_text_column_name: str,
        target_column_name: str,
        datasets_path: Path = DATASET_PATH,
    ):
        super().__init__()

        self.datasets_path = datasets_path
        self.input_text_column_name = input_text_column_name
        self.target_column_name = target_column_name

    def load(self, dataset: HuggingFaceDataset[Data]) -> Corpus:
        output_path = self.datasets_path.joinpath(dataset.dataset)

        flair_datasets = self._get_flair_datasets(dataset, output_path)

        return self._to_flair_corpus(dataset, flair_datasets)

    def _get_flair_datasets(
        self, dataset: HuggingFaceDataset[Data], output_path: Path
    ) -> Dict[str, CSVClassificationDataset]:
        if not all_files_exists(output_path, files=[x + ".csv" for x in self.HUGGING_FACE_SUBSETS]):
            output_path.mkdir(parents=True, exist_ok=True)
            flair_datasets = self._preprocess(dataset, output_path)
        else:
            flair_datasets = self._load(output_path)
        return flair_datasets

    @staticmethod
    def _to_flair_corpus(
        dataset: HuggingFaceDataset[Data], flair_datasets: Dict[str, CSVClassificationDataset]
    ) -> Corpus:
        if not flair_datasets["train"]:
            raise ValueError(
                f"Hugging Face dataset {dataset.dataset} does not have " f"TRAIN subset."
            )

        return Corpus(
            train=flair_datasets["train"],
            dev=flair_datasets["validation"],
            test=flair_datasets["test"],
            name=dataset.dataset,
        )

    def _preprocess(
        self, dataset: HuggingFaceDataset, output_path: Path
    ) -> Dict[str, CSVClassificationDataset]:
        flair_datasets = {}

        hf_dataset = datasets.load_dataset(dataset.dataset, **dataset.load_dataset_kwargs)
        self._log_info(dataset, hf_dataset)

        for subset_name in hf_dataset.keys():
            self._check_compatibility(hf_dataset[subset_name])

        for subset_name in self.HUGGING_FACE_SUBSETS:
            if subset_name in hf_dataset.keys():
                flair_datasets[subset_name] = self._preprocess_subset(
                    hf_dataset, subset_name, output_path
                )
            else:
                flair_datasets[subset_name] = None
        return flair_datasets

    def _log_info(
        self, dataset: HuggingFaceDataset[Data], hf_dataset: datasets.DatasetDict
    ) -> None:
        self._logger.info(f"Dataset name: {dataset.dataset}")

        subsets_info = {
            subset: pprint.pformat(hf_dataset[subset].info.__dict__) for subset in hf_dataset.keys()
        }
        for k, v in groupby(subsets_info.items(), itemgetter(1)):
            self._logger.info(f"Info of {list(map(itemgetter(0), v))}:\n{k}")
        self._logger.info(f"Schemas:\t{hf_dataset}")

    def _preprocess_subset(
        self, hf_dataset: datasets.DatasetDict, subset_name: str, output_path: Path
    ) -> CSVClassificationDataset:
        label_map = hf_dataset[subset_name].features[self.target_column_name].names
        hf_dataset[subset_name] = hf_dataset[subset_name].map(
            lambda row: {"named_target": label_map[row[self.target_column_name]]},
            remove_columns=[self.target_column_name],
        )

        hf_dataset[subset_name].to_csv(
            os.path.join(output_path, f"{subset_name}.csv"), header=False, index=False
        )

        column_name_map = {
            hf_dataset[subset_name].column_names.index(self.input_text_column_name): "text",
            hf_dataset[subset_name].column_names.index("named_target"): "label",
        }

        srsly.write_json(
            output_path.joinpath(f"{subset_name}_column_name_map.json"), column_name_map
        )

        return CSVClassificationDataset(output_path.joinpath(f"{subset_name}.csv"), column_name_map)

    @staticmethod
    def _check_column_in_dataset(dataset: datasets.Dataset, column_name: str) -> None:
        if column_name not in dataset.features:
            raise KeyError(f"Column '{column_name}' not found in features.")

    def _check_classification_task(self, dataset: datasets.Dataset) -> None:
        if not isinstance(dataset.features[self.target_column_name], datasets.ClassLabel):
            raise ValueError(f"Type of target column is not '{datasets.ClassLabel.__name__}'.")

    def _check_compatibility(self, dataset: datasets.Dataset) -> None:
        self._check_column_in_dataset(dataset, self.input_text_column_name)
        self._check_column_in_dataset(dataset, self.target_column_name)
        self._check_classification_task(dataset)

    def _load(self, output_path: Path) -> Dict[str, CSVClassificationDataset]:
        flair_datasets = {}
        for subset_name in self.HUGGING_FACE_SUBSETS:
            column_name_map = srsly.read_json(
                output_path.joinpath(f"{subset_name}_column_name_map.json")
            )
            column_name_map = {int(k): v for k, v in column_name_map.items()}
            flair_datasets[subset_name] = CSVClassificationDataset(
                output_path.joinpath(f"{subset_name}.csv"), column_name_map
            )
        return flair_datasets
