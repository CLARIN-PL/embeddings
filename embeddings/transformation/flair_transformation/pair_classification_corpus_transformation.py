from pathlib import Path
from typing import Tuple

import datasets
from flair.datasets import DataPairDataset

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

        columns = [
            hf_datadict[subset_name].column_names.index(self.input_column_name),
            hf_datadict[subset_name].column_names.index(self.pair_column_name),
            hf_datadict[subset_name].column_names.index(self.target_column_name),
        ]

        return DataPairDataset(
            output_path.joinpath(f"{subset_name}.csv"), columns=columns, separator=","
        )

        # todo: uncomment that after a new flair release for code coherence
        # column_name_map = {
        #     hf_datadict[subset_name].column_names.index(self.input_column_name): "text",
        #     hf_datadict[subset_name].column_names.index(self.pair_column_name): "pair",
        #     hf_datadict[subset_name].column_names.index(self.target_column_name): "label",
        # }
        #
        # return CSVClassificationDataset(output_path.joinpath(f"{subset_name}.csv"),
        #                                 column_name_map)

    def _check_compatibility(self, dataset: datasets.Dataset) -> None:
        super()._check_compatibility(dataset)
        self._check_column_in_dataset(dataset, self.pair_column_name)
