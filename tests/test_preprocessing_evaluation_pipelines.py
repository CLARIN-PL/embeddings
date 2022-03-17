from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import datasets
import flair
import numpy as np
import pytest
import torch
from flair.data import Corpus
from numpy import typing as nptyping

from embeddings.data.data_loader import HuggingFaceDataLoader
from embeddings.data.dataset import LoadableDataset
from embeddings.pipeline.evaluation_pipeline import (
    FlairSequenceLabelingEvaluationPipeline,
    ModelEvaluationPipeline,
)
from embeddings.pipeline.preprocessing_pipeline import PreprocessingPipeline
from embeddings.transformation.flair_transformation.column_corpus_transformation import (
    ColumnCorpusTransformation,
)
from embeddings.transformation.flair_transformation.downsample_corpus_transformation import (
    DownsampleFlairCorpusTransformation,
)
from embeddings.transformation.flair_transformation.split_sample_corpus_transformation import (
    SampleSplitsFlairCorpusTransformation,
)
from embeddings.utils.flair_corpus_persister import FlairConllPersister


@pytest.fixture(scope="module")
def result_path() -> "TemporaryDirectory[str]":
    return TemporaryDirectory()


@pytest.fixture(scope="module")
def embedding_name() -> str:
    return "allegro/herbert-base-cased"


@pytest.fixture(scope="module")
def ner_dataset_name() -> str:
    return "clarin-pl/kpwr-ner"


@pytest.fixture(scope="module")
def hidden_size() -> int:
    return 256


@pytest.fixture(scope="module")
def task_train_kwargs() -> Dict[str, int]:
    return {"max_epochs": 1, "mini_batch_size": 256}


@pytest.fixture(scope="module")
def sequence_labeling_preprocessing_pipeline(
    result_path: "TemporaryDirectory[str]",
    ner_dataset_name: str,
) -> Tuple[PreprocessingPipeline[str, datasets.DatasetDict, Corpus], "TemporaryDirectory[str]"]:
    dataset = LoadableDataset(ner_dataset_name)
    data_loader = HuggingFaceDataLoader()
    transformation = (
        ColumnCorpusTransformation("tokens", "ner")
        .then(SampleSplitsFlairCorpusTransformation(dev_fraction=0.1, test_fraction=0.1, seed=441))
        .then(DownsampleFlairCorpusTransformation(*(0.005, 0.005, 0.005)))
        .persisting(FlairConllPersister(result_path.name))
    )
    pipeline = PreprocessingPipeline(
        dataset=dataset, data_loader=data_loader, transformation=transformation
    )
    return pipeline, result_path


@pytest.fixture(scope="module")
def sequence_labeling_evaluation_pipeline(
    result_path: "TemporaryDirectory[str]",
    embedding_name: str,
    ner_dataset_name: str,
    hidden_size: int,
    task_train_kwargs: Dict[str, int],
) -> Tuple[
    ModelEvaluationPipeline[str, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]],
    "TemporaryDirectory[str]",
]:

    pipeline = FlairSequenceLabelingEvaluationPipeline(
        dataset_path=result_path.name,
        embedding_name=embedding_name,
        output_path=result_path.name,
        hidden_size=hidden_size,
        persist_path=None,
        task_train_kwargs=task_train_kwargs,
    )
    return pipeline, result_path


def test_sequence_labeling_preprocessing_pipeline(
    result_path: "TemporaryDirectory[str]",
    ner_dataset_name: str,
    hidden_size: int,
    task_train_kwargs: Dict[str, int],
    sequence_labeling_preprocessing_pipeline: Tuple[
        PreprocessingPipeline[str, datasets.DatasetDict, Corpus], "TemporaryDirectory[str]"
    ],
    sequence_labeling_evaluation_pipeline: Tuple[
        ModelEvaluationPipeline[str, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]],
        "TemporaryDirectory[str]",
    ],
) -> None:
    flair.set_seed(441)
    flair.device = torch.device("cpu")

    preprocessing_pipeline, path = sequence_labeling_preprocessing_pipeline
    preprocessing_pipeline.run()
    evaluation_pipeline, _ = sequence_labeling_evaluation_pipeline
    result = evaluation_pipeline.run()

    np.testing.assert_almost_equal(
        result["seqeval__mode_None__scheme_None"]["overall_accuracy"], 0.7881773
    )
    np.testing.assert_almost_equal(result["seqeval__mode_None__scheme_None"]["overall_f1"], 0)

    path.cleanup()
