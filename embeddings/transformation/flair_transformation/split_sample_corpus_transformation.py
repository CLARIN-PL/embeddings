from typing import Optional

from flair.data import Corpus, FlairDataset

from embeddings.transformation.flair_transformation.corpus_sampling_transformation import (
    CorpusSamplingTransformation,
)


class SampleSplitsFlairCorpusTransformation(CorpusSamplingTransformation):
    def __init__(
        self,
        dev_fraction: Optional[float] = 0.2,
        test_fraction: Optional[float] = 0.1,
        stratify: bool = True,
        seed: int = 441,
    ):
        super().__init__(stratify, seed)
        dev_fraction = dev_fraction if dev_fraction else 0.0
        test_fraction = test_fraction if test_fraction else 0.0
        assert 1 > dev_fraction >= 0
        assert 1 > test_fraction >= 0
        self.dev_fraction: float = dev_fraction
        self.test_fraction: float = test_fraction
        self.seed: int = seed

    def get_held_out_fraction(self, data: Corpus) -> float:
        if data.dev and not data.test:
            fraction = self.test_fraction
        elif data.test and not data.dev:
            fraction = self.dev_fraction
        elif not data.dev and not data.test:
            fraction = self.dev_fraction + self.test_fraction
        else:
            fraction = 0.0
        return fraction

    def transform(self, data: Corpus) -> Corpus:
        train_subset: FlairDataset
        held_out_subset: FlairDataset
        dev_subset: Optional[FlairDataset]
        test_subset: Optional[FlairDataset]

        held_out_fraction = self.get_held_out_fraction(data)
        if held_out_fraction == 0:
            return data

        train_subset, held_out_subset = self.randomly_split_into_two_datasets(
            dataset=data.train, fraction_size=held_out_fraction
        )

        if (
            data.dev is None
            and data.test is None
            and self.dev_fraction > 0
            and self.test_fraction > 0
        ):
            dev_subset, test_subset = self.randomly_split_into_two_datasets(
                dataset=held_out_subset,
                fraction_size=(self.test_fraction / held_out_fraction),
            )
            data = Corpus(train=train_subset, dev=dev_subset, test=test_subset)
        elif data.test is None and self.test_fraction > 0:
            data = Corpus(train=train_subset, dev=data.dev, test=held_out_subset)
        elif data.dev is None and self.dev_fraction > 0:
            data = Corpus(train=train_subset, dev=held_out_subset, test=data.test)
        else:
            raise ValueError("Unrecognized configuration of dev or test fraction for given Corpus.")

        return data
