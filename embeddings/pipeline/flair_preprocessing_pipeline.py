import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import datasets
from flair.data import Corpus

from embeddings.data.data_loader import (
    FLAIR_DATALOADERS,
    ConllFlairCorpusDataLoader,
    PickleFlairCorpusDataLoader,
    get_flair_dataloader,
)
from embeddings.data.dataset import Dataset
from embeddings.defaults import DATASET_PATH
from embeddings.pipeline import (
    DOWNSAMPLE_SPLITS_TYPE,
    FLAIR_DATASET_TRANSFORMATIONS_TYPE,
    FLAIR_PERSISTERS_TYPE,
    SAMPLE_MISSING_SPLITS_TYPE,
)
from embeddings.pipeline.preprocessing_pipeline import PreprocessingPipeline
from embeddings.transformation.flair_transformation.classification_corpus_transformation import (
    ClassificationCorpusTransformation,
)
from embeddings.transformation.flair_transformation.column_corpus_transformation import (
    ColumnCorpusTransformation,
)
from embeddings.transformation.flair_transformation.downsample_corpus_transformation import (
    DownsampleFlairCorpusTransformation,
)
from embeddings.transformation.flair_transformation.drop_subset_corpus_transformation import (
    DropSubsetFlairCorpusTransformation,
)
from embeddings.transformation.flair_transformation.pair_classification_corpus_transformation import (
    PairClassificationCorpusTransformation,
)
from embeddings.transformation.flair_transformation.split_sample_corpus_transformation import (
    SampleSplitsFlairCorpusTransformation,
)
from embeddings.transformation.transformation import DummyTransformation
from embeddings.utils.flair_corpus_persister import FlairConllPersister, FlairPicklePersister


# Mypy currently properly don't handle dataclasses with abstract methods  https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class FlairPreprocessingPipeline(
    PreprocessingPipeline[str, Union[datasets.DatasetDict, Corpus], Corpus],
    abc.ABC,
):
    persister: FLAIR_PERSISTERS_TYPE = field(init=False)

    dataset_name_or_path: str
    persist_path: str
    input_column_name: Union[str, Tuple[str, str]]
    target_column_name: str
    datasets_path: Path = DATASET_PATH
    sample_missing_splits: Optional[SAMPLE_MISSING_SPLITS_TYPE] = None
    downsample_splits: Optional[DOWNSAMPLE_SPLITS_TYPE] = None
    downsample_splits_stratification: bool = True
    ignore_test_subset: bool = False
    seed: int = 441
    load_dataset_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.persister = self._get_persister()
        dataset = self._get_dataset()
        data_loader = self._get_dataloader(dataset)
        transformation = self._get_transformations(data_loader)
        super(FlairPreprocessingPipeline, self).__init__(dataset, data_loader, transformation)

    @abc.abstractmethod
    def _get_base_dataset_transformation(self) -> FLAIR_DATASET_TRANSFORMATIONS_TYPE:
        pass

    @abc.abstractmethod
    def _get_persister(self) -> FLAIR_PERSISTERS_TYPE:
        pass

    def _get_dataset(self) -> Dataset:
        return Dataset(
            self.dataset_name_or_path,
            **self.load_dataset_kwargs if self.load_dataset_kwargs else {}
        )

    def _get_dataloader(self, dataset: Dataset) -> FLAIR_DATALOADERS:
        return get_flair_dataloader(dataset)

    def _get_dataset_transformation(
        self, data_loader: FLAIR_DATALOADERS
    ) -> FLAIR_DATASET_TRANSFORMATIONS_TYPE:
        if isinstance(data_loader, (ConllFlairCorpusDataLoader, PickleFlairCorpusDataLoader)):
            return DummyTransformation()

        return self._get_base_dataset_transformation()

    def _get_transformations(
        self, data_loader: FLAIR_DATALOADERS
    ) -> FLAIR_DATASET_TRANSFORMATIONS_TYPE:

        transformation = self._get_dataset_transformation(data_loader)

        if self.sample_missing_splits:
            transformation = transformation.then(
                SampleSplitsFlairCorpusTransformation(*self.sample_missing_splits, seed=self.seed)
            )

        if self.ignore_test_subset:
            transformation = transformation.then(DropSubsetFlairCorpusTransformation(subset="test"))

        if self.downsample_splits:
            transformation = transformation.then(
                DownsampleFlairCorpusTransformation(
                    *self.downsample_splits,
                    stratify=self.downsample_splits_stratification,
                    seed=self.seed
                )
            )

        transformation = transformation.persisting(self.persister)
        return transformation


class FlairTextClassificationPreprocessingPipeline(FlairPreprocessingPipeline):
    def _get_persister(self) -> FLAIR_PERSISTERS_TYPE:
        return FlairPicklePersister(self.persist_path)

    def _get_base_dataset_transformation(self) -> FLAIR_DATASET_TRANSFORMATIONS_TYPE:
        assert isinstance(self.input_column_name, str)
        return ClassificationCorpusTransformation(
            input_column_name=self.input_column_name,
            target_column_name=self.target_column_name,
            datasets_path=self.datasets_path,
        )


class FlairTextPairClassificationPreprocessingPipeline(
    FlairTextClassificationPreprocessingPipeline
):
    def _get_base_dataset_transformation(self) -> FLAIR_DATASET_TRANSFORMATIONS_TYPE:
        assert isinstance(self.input_column_name, (tuple, list))
        return PairClassificationCorpusTransformation(
            input_columns_names_pair=self.input_column_name,
            target_column_name=self.target_column_name,
            datasets_path=self.datasets_path,
        )


class FlairSequenceLabelingPreprocessingPipeline(FlairPreprocessingPipeline):
    def _get_persister(self) -> FLAIR_PERSISTERS_TYPE:
        return FlairConllPersister(self.persist_path)

    def _get_base_dataset_transformation(self) -> FLAIR_DATASET_TRANSFORMATIONS_TYPE:
        assert isinstance(self.input_column_name, str)
        return ColumnCorpusTransformation(
            input_column_name=self.input_column_name,
            target_column_name=self.target_column_name,
            datasets_path=self.datasets_path,
        )
