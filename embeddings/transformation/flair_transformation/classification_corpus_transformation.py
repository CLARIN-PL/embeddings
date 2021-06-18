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
        label_map = hf_datadict[subset_name].features[self.target_column_name].names
        hf_datadict[subset_name] = hf_datadict[subset_name].map(
            lambda row: {self.target_column_name: label_map[row[self.target_column_name]]},
        )

        hf_datadict[subset_name].to_csv(
            output_path.joinpath(f"{subset_name}.csv"), header=False, index=False
        )

        column_name_map = {
            hf_datadict[subset_name].column_names.index(self.input_column_name): "text",
            hf_datadict[subset_name].column_names.index(self.target_column_name): "label",
        }

        return CSVClassificationDataset(output_path.joinpath(f"{subset_name}.csv"), column_name_map)

    def _check_task(self, dataset: datasets.Dataset) -> None:
        if not isinstance(dataset.features[self.target_column_name], datasets.ClassLabel):
            raise ValueError(f"Type of target column is not '{datasets.ClassLabel.__name__}'.")
