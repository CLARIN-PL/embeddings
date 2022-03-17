from typing import Optional, Union

from flair.data import Corpus, FlairDataset
from torch.utils.data import Subset

from embeddings.transformation.flair_transformation.corpus_sampling_transformation import (
    CorpusSamplingTransformation,
)


class DownsampleFlairCorpusTransformation(CorpusSamplingTransformation):
    def __init__(
        self,
        downsample_train: Optional[float] = None,
        downsample_dev: Optional[float] = None,
        downsample_test: Optional[float] = None,
        stratify: bool = True,
        seed: int = 441,
    ):
        super().__init__(stratify, seed)
        self.downsample_train = downsample_train
        self.downsample_dev = downsample_dev
        self.downsample_test = downsample_test
        self.seed = seed

    def _downsample_subset(
        self, data: FlairDataset, percentage: Optional[float]
    ) -> Union[FlairDataset, Subset[FlairDataset]]:
        if data and percentage:
            data = self._downsample_to_proportion(data, percentage)
        return data

    def _downsample_to_proportion(
        self, dataset: FlairDataset, percentage: float
    ) -> Subset[FlairDataset]:
        _, downsampled_dataset = self.randomly_split_into_two_datasets(
            dataset=dataset, fraction_size=percentage
        )
        return downsampled_dataset

    def transform(self, data: Corpus) -> Corpus:
        train: FlairDataset = self._downsample_subset(data.train, self.downsample_train)
        dev: FlairDataset = self._downsample_subset(data.dev, self.downsample_dev)
        test: FlairDataset = self._downsample_subset(data.test, self.downsample_test)
        return Corpus(train=train, dev=dev, test=test, sample_missing_splits=False)
