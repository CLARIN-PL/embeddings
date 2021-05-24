import os
import pprint
import tempfile
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Dict

import datasets
import srsly
from flair.data import Corpus
from flair.datasets import CSVClassificationDataset

from embeddings.transformation.transformation import Transformation
from experimental.defaults import DATASET_PATH


class ToFlairCorpusTransformation(Transformation[datasets.DatasetDict, Corpus]):
    HUGGING_FACE_SUBSETS = ["train", "validation", "test"]

    def __init__(
        self,
        input_text_column_name: str,
        target_column_name: str,
        datasets_path: Path = DATASET_PATH,
    ):
        super().__init__()

        self.datasets_cache_path = datasets_path
        self.input_text_column_name = input_text_column_name
        self.target_column_name = target_column_name

    def transform(self, data: datasets.DatasetDict) -> Corpus:
        with tempfile.TemporaryDirectory() as tmp_dir_path:
            output_path = Path(tmp_dir_path)
            flair_datasets = self._preprocess(data, output_path)
        return self._to_flair_corpus(flair_datasets)

    @staticmethod
    def _to_flair_corpus(flair_datasets: Dict[str, CSVClassificationDataset]) -> Corpus:
        if not flair_datasets["train"]:
            raise ValueError(f"Hugging Face dataset does not contain TRAIN subset.")

        return Corpus(
            train=flair_datasets["train"],
            dev=flair_datasets["validation"],
            test=flair_datasets["test"],
        )

    def _preprocess(
        self, hf_datadict: datasets.DatasetDict, output_path: Path
    ) -> Dict[str, CSVClassificationDataset]:
        flair_datasets = {}
        self._log_info(hf_datadict)

        for subset_name in hf_datadict.keys():
            self._check_compatibility(hf_datadict[subset_name])

        for subset_name in self.HUGGING_FACE_SUBSETS:
            if subset_name in hf_datadict.keys():
                flair_datasets[subset_name] = self._preprocess_subset(
                    hf_datadict, subset_name, output_path
                )
            else:
                flair_datasets[subset_name] = None
        return flair_datasets

    def _log_info(self, hf_datadict: datasets.DatasetDict) -> None:
        subsets_info = {
            subset: pprint.pformat(hf_datadict[subset].info.__dict__)
            for subset in hf_datadict.keys()
        }
        for k, v in groupby(subsets_info.items(), itemgetter(1)):
            self._logger.info(f"Info of {list(map(itemgetter(0), v))}:\n{k}")
        self._logger.info(f"Schemas:\t{hf_datadict}")

    def _preprocess_subset(
        self, hf_datadict: datasets.DatasetDict, subset_name: str, output_path: Path
    ) -> CSVClassificationDataset:
        label_map = hf_datadict[subset_name].features[self.target_column_name].names
        hf_datadict[subset_name] = hf_datadict[subset_name].map(
            lambda row: {"named_target": label_map[row[self.target_column_name]]},
            remove_columns=[self.target_column_name],
        )

        hf_datadict[subset_name].to_csv(
            os.path.join(output_path, f"{subset_name}.csv"), header=False, index=False
        )

        column_name_map = {
            hf_datadict[subset_name].column_names.index(self.input_text_column_name): "text",
            hf_datadict[subset_name].column_names.index("named_target"): "label",
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


class DownsampleFlairCorpusTransformation(Transformation[Corpus, Corpus]):
    def __init__(
        self,
        percentage: float,
        downsample_train: bool = True,
        downsample_dev: bool = True,
        downsample_test: bool = True,
    ):
        self.percentage = percentage
        self.downsample_train = downsample_train
        self.downsample_dev = downsample_dev
        self.downsample_test = downsample_test

    def transform(self, data: Corpus) -> Corpus:
        return data.downsample(
            percentage=self.percentage,
            downsample_train=self.downsample_train,
            downsample_dev=self.downsample_dev,
            downsample_test=self.downsample_test,
        )
