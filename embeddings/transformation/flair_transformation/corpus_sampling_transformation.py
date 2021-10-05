from abc import ABC
from typing import Tuple, Union

from flair.data import Corpus, FlairDataset
from flair.datasets import CSVClassificationDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from embeddings.transformation.transformation import Transformation


class CorpusSamplingTransformation(Transformation[Corpus, Corpus], ABC):
    def __init__(self, stratify: bool = True, seed: int = 441):
        self.stratify = stratify
        self.seed = seed

    def randomly_split_into_two_datasets(
        self,
        dataset: Union[FlairDataset, Subset[FlairDataset]],
        fraction_size: float,
    ) -> Tuple[Subset[FlairDataset], Subset[FlairDataset]]:
        if isinstance(dataset, Subset):
            indices = dataset.indices
            dataset = dataset.dataset
        else:
            indices = range(len(dataset))

        labels = None
        if self.stratify and isinstance(dataset, CSVClassificationDataset):
            labels = [sentence.labels[0].value for sentence in dataset]

        first_indices, second_indices = train_test_split(
            indices, test_size=fraction_size, shuffle=True, random_state=self.seed, stratify=labels
        )
        first_indices.sort()
        second_indices.sort()

        return Subset(dataset, first_indices), Subset(dataset, second_indices)
