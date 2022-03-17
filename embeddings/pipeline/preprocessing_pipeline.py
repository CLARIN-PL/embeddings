import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generic, Optional, Tuple, Type, Union

import datasets
from flair.data import Corpus

from embeddings.data.data_loader import (
    FLAIR_DATALOADERS,
    ConllFlairCorpusDataLoader,
    DataLoader,
    PickleFlairCorpusDataLoader,
    get_flair_dataloader,
)
from embeddings.data.dataset import Data, Dataset, LoadableDataset
from embeddings.defaults import DATASET_PATH
from embeddings.pipeline.pipeline import Pipeline
from embeddings.pipeline.standard_pipeline import LoaderResult, TransformationResult
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
from embeddings.transformation.transformation import DummyTransformation, Transformation
from embeddings.utils.flair_corpus_persister import FlairConllPersister, FlairPicklePersister


class PreprocessingPipeline(
    Pipeline[TransformationResult], Generic[Data, LoaderResult, TransformationResult]
):
    def __init__(
        self,
        dataset: Dataset[Data],
        data_loader: DataLoader[Data, LoaderResult],
        transformation: Transformation[LoaderResult, TransformationResult],
    ) -> None:
        self.dataset = dataset
        self.data_loader = data_loader
        self.transformation = transformation

    def run(self) -> TransformationResult:
        loaded_data = self.data_loader.load(self.dataset)
        result = self.transformation.transform(loaded_data)
        return result


# Mypy currently properly don't handle dataclasses with abstract methods  https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class FlairPreprocessingPipeline(
    PreprocessingPipeline[str, Union[datasets.DatasetDict, Corpus], Corpus],
    abc.ABC,
):
    corpus_transformation_cls: Type[
        Union[
            ClassificationCorpusTransformation,
            PairClassificationCorpusTransformation,
            ColumnCorpusTransformation,
        ]
    ] = field(init=False)

    persister: Union[FlairConllPersister[Corpus], FlairPicklePersister[Corpus, Corpus]] = field(
        init=False
    )

    dataset_name_or_path: str
    persist_path: str
    input_column_name: Union[str, Tuple[str, str]]
    target_column_name: str
    datasets_path: Path = DATASET_PATH
    sample_missing_splits: Optional[Tuple[Optional[float], Optional[float]]] = None
    downsample_splits: Optional[Tuple[Optional[float], Optional[float], Optional[float]]] = None
    downsample_splits_stratification: bool = True
    ignore_test_subset: bool = False
    seed: int = 441
    load_dataset_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self._setup()
        dataset, data_loader, transformation = self._setup_transformations()
        super(FlairPreprocessingPipeline, self).__init__(dataset, data_loader, transformation)

    @abc.abstractmethod
    def _setup(self) -> None:
        pass

    def _setup_transformations(
        self,
    ) -> Tuple[
        LoadableDataset,
        FLAIR_DATALOADERS,
        Union[Transformation[datasets.DatasetDict, Corpus], Transformation[Corpus, Corpus]],
    ]:
        dataset = LoadableDataset(
            self.dataset_name_or_path,
            **self.load_dataset_kwargs if self.load_dataset_kwargs else {}
        )
        data_loader: FLAIR_DATALOADERS = get_flair_dataloader(dataset)

        transformation: Union[
            Transformation[datasets.DatasetDict, Corpus], Transformation[Corpus, Corpus]
        ] = DummyTransformation()

        if not isinstance(data_loader, (ConllFlairCorpusDataLoader, PickleFlairCorpusDataLoader)):
            if self.corpus_transformation_cls == ClassificationCorpusTransformation:
                assert isinstance(self.input_column_name, str)
                transformation = ClassificationCorpusTransformation(
                    input_column_name=self.input_column_name,
                    target_column_name=self.target_column_name,
                    datasets_path=self.datasets_path,
                )
            elif self.corpus_transformation_cls == ColumnCorpusTransformation:
                assert isinstance(self.input_column_name, str)
                transformation = ColumnCorpusTransformation(
                    input_column_name=self.input_column_name,
                    target_column_name=self.target_column_name,
                    datasets_path=self.datasets_path,
                )

            elif self.corpus_transformation_cls == PairClassificationCorpusTransformation:
                assert isinstance(self.input_column_name, tuple)
                transformation = PairClassificationCorpusTransformation(
                    input_columns_names_pair=self.input_column_name,
                    target_column_name=self.target_column_name,
                    datasets_path=self.datasets_path,
                )
            else:
                raise ValueError(
                    "Unrecognized corpus transformation cls. "
                    "Try ClassificationCorpusTransformation, ColumnCorpusTransformation or PairClassificationCorpusTransformation"
                )

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

        return dataset, data_loader, transformation


class FlairTextClassificationPreprocessingPipeline(FlairPreprocessingPipeline):
    def _setup(self) -> None:
        self.persister = FlairPicklePersister(self.persist_path)
        self.corpus_transformation_cls = ClassificationCorpusTransformation


class FlairTextPairClassificationPreprocessingPipeline(FlairPreprocessingPipeline):
    def _setup(self) -> None:
        self.persister = FlairPicklePersister(self.persist_path)
        self.corpus_transformation_cls = PairClassificationCorpusTransformation


class FlairSequenceLabelingPreprocessingPipeline(FlairPreprocessingPipeline):
    def _setup(self) -> None:
        self.persister = FlairConllPersister(self.persist_path)
        self.corpus_transformation_cls = ColumnCorpusTransformation
