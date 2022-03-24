from typing import Optional, Tuple, Union

import datasets
from flair.data import Corpus

from embeddings.transformation.transformation import Transformation
from embeddings.utils.flair_corpus_persister import FlairConllPersister, FlairPicklePersister

DOWNSAMPLE_SPLITS_TYPE = Tuple[Optional[float], Optional[float], Optional[float]]
SAMPLE_MISSING_SPLITS_TYPE = Optional[Tuple[Optional[float], Optional[float]]]
FLAIR_DATASET_TRANSFORMATIONS_TYPE = Union[
    Transformation[datasets.DatasetDict, Corpus], Transformation[Corpus, Corpus]
]
FLAIR_PERSISTERS_TYPE = Union[FlairConllPersister[Corpus], FlairPicklePersister[Corpus, Corpus]]
