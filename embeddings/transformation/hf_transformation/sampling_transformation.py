import abc
from typing import Optional

import datasets
import sklearn.model_selection
from datasets import DatasetDict

from embeddings.transformation.transformation import Transformation
from embeddings.utils.loggers import get_logger

_logger = get_logger(__name__)


class SampleSplitsTransformation(Transformation[datasets.DatasetDict, datasets.DatasetDict]):
    def __init__(
        self,
        dev_fraction: Optional[float] = None,
        test_fraction: Optional[float] = None,
        seed: int = 441,
    ):
        self.dev_fraction = dev_fraction
        self.test_fraction = test_fraction
        self.seed = seed

        if not dev_fraction and not test_fraction:
            raise ValueError(
                "At least one of parameters `dev_fraction` and `test_fraction` must be set for SampleSplitsHuggingFaceTransformation"
            )

    @abc.abstractmethod
    def _train_test_split(
        self, data: datasets.Dataset, test_fraction: float
    ) -> datasets.DatasetDict:
        pass

    def _check_args(self, data: datasets.DatasetDict) -> None:
        if "test" in data.keys() and self.test_fraction:
            _logger.warning(
                "Original test subset found in dataset keys, therefore won't be replaced."
                "Use DropSubsetTransformation firstly in case if you want to replace original test subset"
            )
        if "validation" in data.keys() and self.dev_fraction:
            _logger.warning(
                "Original validation subset found in dataset keys, therefore won't be replaced."
                "Use DropSubsetTransformation firstly in case if you want to replace original validation subset"
            )

    def transform(self, data: datasets.DatasetDict) -> datasets.DatasetDict:
        dataset = datasets.DatasetDict()
        self._check_args(data)

        if (
            self.dev_fraction
            and self.test_fraction
            and "validation" not in data.keys()
            and "test" not in data.keys()
        ):
            dev_test_fraction = self.dev_fraction + self.test_fraction
            held_out_dataset = self._train_test_split(data["train"], dev_test_fraction)
            dataset["train"] = held_out_dataset["train"]
            test_valid_dataset = self._train_test_split(
                held_out_dataset["test"], self.test_fraction / dev_test_fraction
            )
            dataset["validation"] = test_valid_dataset["train"]
            dataset["test"] = test_valid_dataset["test"]

        elif self.dev_fraction and "validation" not in data:
            sampled_dataset = self._train_test_split(data["train"], self.dev_fraction)
            dataset["train"] = sampled_dataset["train"]
            dataset["validation"] = sampled_dataset["test"]
            if "test" in data:
                dataset["test"] = data["test"]

        elif self.test_fraction and "test" not in data:
            dataset = self._train_test_split(data["train"], self.test_fraction)
            if "validation" in data:
                dataset["validation"] = data["validation"]
        else:
            _logger.warning(
                "Sampling transformation wrongly defined. "
                "Subsets can not be overwritten. Returning original dataset"
            )
            return data

        return dataset


class SampleSplitsHuggingFaceTransformation(SampleSplitsTransformation):
    def _train_test_split(
        self, data: datasets.Dataset, test_fraction: float
    ) -> datasets.DatasetDict:
        return data.train_test_split(test_size=test_fraction, seed=self.seed)  # type: ignore


class SampleSplitsStratifiedTransformation(SampleSplitsTransformation):
    def __init__(
        self,
        target_field_name: str,
        dev_fraction: Optional[float] = None,
        test_fraction: Optional[float] = None,
        seed: int = 441,
    ):
        super().__init__(dev_fraction, test_fraction, seed)
        self.target_field_name = target_field_name

    def _train_test_split(
        self, data: datasets.Dataset, test_fraction: float
    ) -> datasets.DatasetDict:
        data_idx = range(len(data))

        train_idx, test_idx = sklearn.model_selection.train_test_split(
            data_idx,
            test_size=test_fraction,
            random_state=self.seed,
            stratify=data[self.target_field_name],
        )

        train_split = data.select(train_idx)
        test_split = data.select(test_idx)

        return DatasetDict({"train": train_split, "test": test_split})
