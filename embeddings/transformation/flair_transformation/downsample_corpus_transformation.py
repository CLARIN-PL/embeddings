from typing import Tuple, Union

from flair.data import Corpus, FlairDataset
from torch.utils.data import Subset

from embeddings.transformation.flair_transformation.corpus_sampling_transformation import (
    CorpusSamplingTransformation,
)


class DownsampleFlairCorpusTransformation(CorpusSamplingTransformation):
    def __init__(
        self,
        percentage: float,
        downsample_train: bool = True,
        downsample_dev: bool = True,
        downsample_test: bool = True,
        seed: int = 441,
    ):
        self.percentage = percentage
        self.downsample_train = downsample_train
        self.downsample_dev = downsample_dev
        self.downsample_test = downsample_test
        self.seed = seed

    def _downsample(
        self,
        data: Corpus,
        percentage: float = 0.1,
        downsample_train: bool = True,
        downsample_dev: bool = True,
        downsample_test: bool = True,
    ) -> Tuple[Subset[FlairDataset], Subset[FlairDataset], Subset[FlairDataset]]:
        train = self._downsample_subset(data.train, downsample_train, percentage=percentage)
        dev = self._downsample_subset(data.dev, downsample_dev, percentage=percentage)
        test = self._downsample_subset(data.test, downsample_test, percentage=percentage)
        return train, dev, test

    def _downsample_subset(
        self, data: FlairDataset, downsample: bool, percentage: float
    ) -> Union[FlairDataset, Subset[FlairDataset]]:
        if data and downsample:
            data = self._downsample_to_proportion(data, percentage)
        return data

    def _downsample_to_proportion(
        self, dataset: FlairDataset, percentage: float
    ) -> Subset[FlairDataset]:
        _, downsampled_dataset = self.randomly_split_into_two_datasets(
            dataset=dataset, fraction_size=percentage, random_state=self.seed
        )
        return downsampled_dataset

    def transform(self, data: Corpus) -> Corpus:
        train: FlairDataset
        dev: FlairDataset
        test: FlairDataset
        train, dev, test = self._downsample(
            data,
            percentage=self.percentage,
            downsample_train=self.downsample_train,
            downsample_dev=self.downsample_dev,
            downsample_test=self.downsample_test,
        )
        return Corpus(train=train, dev=dev, test=test, sample_missing_splits=False)
