from abc import ABC
from typing import List, Optional, Tuple, Union

from flair.data import Corpus, FlairDataset
from flair.datasets import CSVClassificationDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from embeddings.transformation.transformation import Transformation
from embeddings.utils.loggers import get_logger

_logger = get_logger(__name__)


class CorpusSamplingTransformation(Transformation[Corpus, Corpus], ABC):
    def __init__(self, stratify: bool = True, seed: int = 441):
        self.stratify = stratify
        self.seed = seed

    @staticmethod
    def retrieve_labels(dataset: FlairDataset) -> List[str]:
        return [sentence.labels[0].value for sentence in dataset]

    def randomly_split_into_two_datasets(
        self,
        dataset: Union[FlairDataset, Subset[FlairDataset]],
        fraction_size: float,
    ) -> Tuple[Subset[FlairDataset], Subset[FlairDataset]]:

        labels: Optional[List[str]] = None
        if isinstance(dataset, Subset):
            indices = dataset.indices
            if self.stratify:
                if isinstance(dataset.dataset, CSVClassificationDataset):
                    labels = self.retrieve_labels(dataset)
                else:
                    _logger.warning(
                        "Stratification for datasets other than CSVClassificationDataset not supported yet"
                    )
            dataset = dataset.dataset
        else:
            indices = range(len(dataset))
            if self.stratify:
                if isinstance(dataset, CSVClassificationDataset):
                    labels = self.retrieve_labels(dataset)
                else:
                    _logger.warning(
                        "Stratification for datasets other than CSVClassificationDataset not supported yet"
                    )

        first_indices, second_indices = train_test_split(
            indices, test_size=fraction_size, shuffle=True, random_state=self.seed, stratify=labels
        )
        first_indices.sort()
        second_indices.sort()

        return Subset(dataset, first_indices), Subset(dataset, second_indices)
