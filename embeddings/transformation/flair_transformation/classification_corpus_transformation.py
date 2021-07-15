from pathlib import Path

import datasets
from flair.datasets import CSVClassificationDataset

from embeddings.transformation.flair_transformation.corpus_transformation import (
    CorpusTransformation,
)


class ClassificationCorpusTransformation(CorpusTransformation):
    def _preprocess_subset(
        self, hf_datadict: datasets.DatasetDict, subset_name: str, output_path: Path
    ) -> CSVClassificationDataset:
        csv_path = output_path.joinpath(f"{subset_name}.csv")
        self._save_to_csv(hf_datadict, subset_name, csv_path)
        column_name_map = {
            hf_datadict[subset_name].column_names.index(self.input_column_name): "text",
            hf_datadict[subset_name].column_names.index(self.target_column_name): "label",
        }
        return CSVClassificationDataset(csv_path, column_name_map)

    def _save_to_csv(
        self, hf_datadict: datasets.DatasetDict, subset_name: str, csv_path: Path
    ) -> None:
        label_map = hf_datadict[subset_name].features[self.target_column_name].names
        hf_datadict[subset_name] = hf_datadict[subset_name].map(
            lambda row: {self.target_column_name: label_map[row[self.target_column_name]]},
        )
        hf_datadict[subset_name].to_csv(csv_path, header=False, index=False)

    def _check_task(self, dataset: datasets.Dataset) -> None:
        if not isinstance(dataset.features[self.target_column_name], datasets.ClassLabel):
            raise ValueError(f"Type of target column is not '{datasets.ClassLabel.__name__}'.")
