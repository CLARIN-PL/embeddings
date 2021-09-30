from abc import ABC
from typing import Tuple, Union

from flair.data import Corpus, FlairDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from embeddings.transformation.transformation import Transformation


class CorpusSamplingTransformation(Transformation[Corpus, Corpus], ABC):
    @staticmethod
    def randomly_split_into_two_datasets(
        dataset: Union[FlairDataset, Subset[FlairDataset]],
        fraction_size: float,
        random_state: int = 441,
    ) -> Tuple[Subset[FlairDataset], Subset[FlairDataset]]:
        if isinstance(dataset, Subset):
            indices = dataset.indices
            dataset = dataset.dataset
        else:
            indices = range(len(dataset))
        first_indices, second_indices = train_test_split(
            indices,
            test_size=fraction_size,
            shuffle=True,
            random_state=random_state,
        )
        first_indices.sort()
        second_indices.sort()
        return Subset(dataset, first_indices), Subset(dataset, second_indices)
