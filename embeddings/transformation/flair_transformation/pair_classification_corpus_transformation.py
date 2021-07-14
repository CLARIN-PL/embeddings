import warnings
from pathlib import Path
from typing import Tuple

import datasets
import flair
from flair.datasets import CSVClassificationDataset, DataPairDataset

from embeddings.defaults import DATASET_PATH
from embeddings.transformation.flair_transformation.classification_corpus_transformation import (
    ClassificationCorpusTransformation,
)


class PairClassificationCorpusTransformation(ClassificationCorpusTransformation):
    def __init__(
        self,
        input_columns_names_pair: Tuple[str, str],
        target_column_name: str,
        datasets_path: Path = DATASET_PATH,
    ):
        super().__init__(input_columns_names_pair[0], target_column_name, datasets_path)
        self.pair_column_name: str = input_columns_names_pair[1]

    def _preprocess_subset(
        self, hf_datadict: datasets.DatasetDict, subset_name: str, output_path: Path
    ) -> DataPairDataset:
        csv_path = output_path.joinpath(f"{subset_name}.csv")
        self._save_to_csv(hf_datadict, subset_name, csv_path)

        if flair.__version__ != "0.8":
            warnings.warn(
                f"Implementation of {type(self).__name__} could be deprecated due to "
                f"new flair release.",
                DeprecationWarning,
            )
        # todo: change that function call for _to_csv_classification_dataset after a new flair
        #  release for code coherence
        return self._to_data_pair_dataset(hf_datadict, subset_name, output_path)

    def _to_data_pair_dataset(
        self, hf_datadict: datasets.DatasetDict, subset_name: str, output_path: Path
    ) -> DataPairDataset:
        columns = [
            hf_datadict[subset_name].column_names.index(self.input_column_name),
            hf_datadict[subset_name].column_names.index(self.pair_column_name),
            hf_datadict[subset_name].column_names.index(self.target_column_name),
        ]

        return DataPairDataset(
            output_path.joinpath(f"{subset_name}.csv"), columns=columns, separator=","
        )

    def _to_csv_classification_dataset(
        self, hf_datadict: datasets.DatasetDict, subset_name: str, output_path: Path
    ) -> CSVClassificationDataset:
        if flair.__version__ == "0.8":
            raise NotImplementedError(
                f"Usage of {CSVClassificationDataset.__name__} for pair "
                f"classification is not possible for flair 0.8."
            )

        column_name_map = {
            hf_datadict[subset_name].column_names.index(self.input_column_name): "text",
            hf_datadict[subset_name].column_names.index(self.pair_column_name): "pair",
            hf_datadict[subset_name].column_names.index(self.target_column_name): "label",
        }

        return CSVClassificationDataset(output_path.joinpath(f"{subset_name}.csv"), column_name_map)

    def _check_compatibility(self, dataset: datasets.Dataset) -> None:
        super()._check_compatibility(dataset)
        self._check_column_in_dataset(dataset, self.pair_column_name)
