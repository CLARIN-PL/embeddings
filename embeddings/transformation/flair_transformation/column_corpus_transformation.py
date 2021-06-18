from pathlib import Path
from typing import Dict

import datasets
from flair.datasets import ColumnDataset

from embeddings.transformation.flair_transformation.corpus_transformation import (
    CorpusTransformation,
)


class ColumnCorpusTransformation(CorpusTransformation):
    def _preprocess_subset(
        self, hf_datadict: datasets.DatasetDict, subset_name: str, output_path: Path
    ) -> ColumnDataset:
        tag_map = hf_datadict[subset_name].features[self.target_column_name].feature.names

        hf_datadict[subset_name] = hf_datadict[subset_name].map(
            lambda row: {
                self.target_column_name: [
                    tag_map[tag_idx] for tag_idx in row[self.target_column_name]
                ]
            },
        )

        column_format = self._save_to_conll(
            hf_datadict[subset_name], output_path.joinpath(f"{subset_name}.csv")
        )

        return ColumnDataset(output_path.joinpath(f"{subset_name}.csv"), column_format)

    def _check_task(self, dataset: datasets.Dataset) -> None:
        if not self._is_input_sequence_of_strings(dataset):
            raise ValueError(
                f"Type of input column is not '{datasets.Sequence.__name__}' of"
                f" '{datasets.Value.__name__}' with strings."
            )

        if not self._is_target_sequence_of_labels(dataset):
            raise ValueError(
                f"Type of target column is not '{datasets.Sequence.__name__}' of"
                f" '{datasets.ClassLabel.__name__}'."
            )

    def _is_input_sequence_of_strings(self, dataset: datasets.Dataset) -> bool:
        if not isinstance(dataset.features[self.input_column_name], datasets.Sequence):
            return False
        elif not isinstance(dataset.features[self.input_column_name].feature, datasets.Value):
            return False
        elif dataset.features[self.input_column_name].feature.dtype != "string":
            return False
        else:
            return True

    def _is_target_sequence_of_labels(self, dataset: datasets.Dataset) -> bool:
        if not isinstance(dataset.features[self.target_column_name], datasets.Sequence):
            return False
        elif not isinstance(dataset.features[self.target_column_name].feature, datasets.ClassLabel):
            return False
        else:
            return True

    def _save_to_conll(self, dataset: datasets.Dataset, output_path: Path) -> Dict[int, str]:
        with open(output_path, "w", encoding="utf-8") as f:
            for tokens, tags in zip(
                dataset[self.input_column_name], dataset[self.target_column_name]
            ):
                assert len(tokens) == len(tags)
                for token, tag in zip(tokens, tags):
                    f.write(f"{token}\t{tag}\n")
                f.write("\n")

        column_format = {0: "text", 1: "tag"}
        return column_format
