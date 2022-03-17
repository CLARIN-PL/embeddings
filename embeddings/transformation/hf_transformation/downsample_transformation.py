from typing import Optional

import datasets

from embeddings.transformation.transformation import Transformation


class DownsampleHuggingFaceTransformation(
    Transformation[datasets.DatasetDict, datasets.DatasetDict]
):
    def __init__(
        self,
        downsample_train: Optional[float] = None,
        downsample_dev: Optional[float] = None,
        downsample_test: Optional[float] = None,
        seed: int = 441,
    ):
        self.downsamples = {
            "train": downsample_train,
            "validation": downsample_dev,
            "test": downsample_test,
        }
        self.seed = seed

    def _downsample_subset(self, data: datasets.Dataset, percentage: float) -> datasets.Dataset:
        downsampled_data = data.train_test_split(percentage, seed=self.seed)["test"]
        assert isinstance(downsampled_data, datasets.Dataset)
        return downsampled_data

    def transform(self, data: datasets.DatasetDict) -> datasets.DatasetDict:
        for column_name, downsample_factor in self.downsamples.items():
            if downsample_factor is not None and column_name in data and 0 < downsample_factor < 1:
                data[column_name] = self._downsample_subset(data[column_name], downsample_factor)

        return data
