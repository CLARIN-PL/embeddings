import datasets

from embeddings.transformation.sampling_transformation import SampleSplitTransformation
from embeddings.utils.loggers import get_logger

_logger = get_logger(__name__)


class SampleSplitsHuggingFaceTransformation(SampleSplitTransformation):
    def _train_test_split(
        self, data: datasets.Dataset, test_fraction: float
    ) -> datasets.DatasetDict:
        return data.train_test_split(test_size=test_fraction, seed=self.seed)  # type: ignore
