from typing import Optional

import datasets

from embeddings.transformation.transformation import Transformation


class SampleSplitsHuggingFaceTransformation(
    Transformation[datasets.DatasetDict, datasets.DatasetDict]
):
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
                "At least one of parameters `dev_fraction` andgi `test_fraction` must be set for SampleSplitsHuggingFaceTransformation"
            )

    def _train_test_split(
        self, data: datasets.Dataset, test_fraction: float
    ) -> datasets.DatasetDict:
        return data.train_test_split(test_size=test_fraction, seed=self.seed)  # type: ignore

    def transform(self, data: datasets.DatasetDict) -> datasets.DatasetDict:
        dataset = datasets.DatasetDict()

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

        elif self.test_fraction and "test" not in data:
            dataset = self._train_test_split(data["train"], self.test_fraction)

        else:
            return data
        return dataset
