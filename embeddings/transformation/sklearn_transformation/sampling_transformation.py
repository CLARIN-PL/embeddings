from typing import Optional

import datasets
import sklearn.model_selection
from datasets import DatasetDict

from embeddings.transformation.sampling_transformation import SampleSplitTransformation
from embeddings.utils.loggers import get_logger

_logger = get_logger(__name__)


class SampleSplitsSklearnTransformation(SampleSplitTransformation):
    def __init__(
        self,
        dev_fraction: Optional[float] = None,
        test_fraction: Optional[float] = None,
        stratify: bool = False,
        target_field_name: Optional[str] = None,
        seed: int = 441,
    ):
        super().__init__(dev_fraction, test_fraction, seed)
        self.stratify = stratify
        self.target_field_name = target_field_name

    def _train_test_split(
        self, data: datasets.Dataset, test_fraction: float
    ) -> datasets.DatasetDict:
        data_idx = range(len(data))

        train_idx, test_idx = sklearn.model_selection.train_test_split(
            data_idx,
            test_size=test_fraction,
            random_state=self.seed,
            stratify=data[self.target_field_name] if self.stratify else None,  # type: ignore
        )

        train_split = data.select(train_idx)
        test_split = data.select(test_idx)

        return DatasetDict({"train": train_split, "test": test_split})
