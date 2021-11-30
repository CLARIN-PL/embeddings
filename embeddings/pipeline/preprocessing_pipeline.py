from pathlib import Path
from typing import Any, Dict, Generic, Optional, Tuple, Union

import datasets
from flair.data import Corpus

from embeddings.data.data_loader import DataLoader, HuggingFaceDataLoader
from embeddings.data.dataset import Data, Dataset, HuggingFaceDataset
from embeddings.defaults import DATASET_PATH
from embeddings.pipeline.pipeline import Pipeline
from embeddings.pipeline.standard_pipeline import LoaderResult, TransformationResult
from embeddings.transformation.flair_transformation.classification_corpus_transformation import (
    ClassificationCorpusTransformation,
)
from embeddings.transformation.flair_transformation.column_corpus_transformation import (
    ColumnCorpusTransformation,
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
from embeddings.transformation.transformation import Transformation
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


class FlairTextClassificationPreprocessingPipeline(
    PreprocessingPipeline[str, datasets.DatasetDict, Corpus]
):
    def __init__(
        self,
        dataset_name: str,
        persist_path: str,
        input_column_name: str,
        target_column_name: str,
        datasets_path: Path = DATASET_PATH,
        sample_missing_splits: Optional[Tuple[Optional[float], Optional[float]]] = None,
        ignore_test_subset: bool = False,
        seed: int = 441,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        dataset = HuggingFaceDataset(
            dataset_name, **load_dataset_kwargs if load_dataset_kwargs else {}
        )
        data_loader = HuggingFaceDataLoader()
        transformation: Union[
            Transformation[datasets.DatasetDict, Corpus], Transformation[Corpus, Corpus]
        ]
        transformation = ClassificationCorpusTransformation(
            input_column_name=input_column_name,
            target_column_name=target_column_name,
            datasets_path=datasets_path,
        )
        if sample_missing_splits:
            transformation = transformation.then(
                SampleSplitsFlairCorpusTransformation(*sample_missing_splits, seed=seed)
            )

        if ignore_test_subset:
            transformation = transformation.then(DropSubsetFlairCorpusTransformation(subset="test"))
        transformation = transformation.persisting(FlairPicklePersister(path=persist_path))
        super().__init__(dataset, data_loader, transformation)


class FlairTextPairClassificationPreprocessingPipeline(
    PreprocessingPipeline[str, datasets.DatasetDict, Corpus]
):
    def __init__(
        self,
        dataset_name: str,
        persist_path: str,
        input_column_names: Tuple[str, str],
        target_column_name: str,
        datasets_path: Path = DATASET_PATH,
        sample_missing_splits: Optional[Tuple[Optional[float], Optional[float]]] = None,
        ignore_test_subset: bool = False,
        seed: int = 441,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        dataset = HuggingFaceDataset(
            dataset_name, **load_dataset_kwargs if load_dataset_kwargs else {}
        )
        data_loader = HuggingFaceDataLoader()
        transformation: Union[
            Transformation[datasets.DatasetDict, Corpus], Transformation[Corpus, Corpus]
        ]
        transformation = PairClassificationCorpusTransformation(
            input_columns_names_pair=input_column_names,
            target_column_name=target_column_name,
            datasets_path=datasets_path,
        )
        if sample_missing_splits:
            transformation = transformation.then(
                SampleSplitsFlairCorpusTransformation(*sample_missing_splits, seed=seed)
            )
        if ignore_test_subset:
            transformation = transformation.then(DropSubsetFlairCorpusTransformation(subset="test"))
        transformation = transformation.persisting(FlairPicklePersister(path=persist_path))
        super().__init__(dataset, data_loader, transformation)


class FlairSequenceLabelingPreprocessingPipeline(
    PreprocessingPipeline[str, datasets.DatasetDict, Corpus]
):
    def __init__(
        self,
        dataset_name: str,
        persist_path: str,
        input_column_name: str,
        target_column_name: str,
        datasets_path: Path = DATASET_PATH,
        sample_missing_splits: Optional[Tuple[Optional[float], Optional[float]]] = None,
        ignore_test_subset: bool = False,
        seed: int = 441,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        dataset = HuggingFaceDataset(
            dataset_name, **load_dataset_kwargs if load_dataset_kwargs else {}
        )
        data_loader = HuggingFaceDataLoader()
        transformation: Union[
            Transformation[datasets.DatasetDict, Corpus], Transformation[Corpus, Corpus]
        ]
        transformation = ColumnCorpusTransformation(
            input_column_name=input_column_name,
            target_column_name=target_column_name,
            datasets_path=datasets_path,
        )
        if sample_missing_splits:
            transformation = transformation.then(
                SampleSplitsFlairCorpusTransformation(*sample_missing_splits, seed=seed)
            )

        if ignore_test_subset:
            transformation = transformation.then(DropSubsetFlairCorpusTransformation(subset="test"))
        transformation = transformation.persisting(FlairConllPersister(path=persist_path))
        super().__init__(dataset, data_loader, transformation)
