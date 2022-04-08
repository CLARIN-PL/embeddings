from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import datasets
import flair
import numpy as np
import pytest
import torch
from flair.data import Corpus
from numpy import typing as nptyping

from embeddings.config.flair_config import (
    FlairSequenceLabelingBasicConfig,
    FlairSequenceLabelingConfig,
)
from embeddings.data.data_loader import HuggingFaceDataLoader
from embeddings.data.dataset import Dataset
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
def default_config() -> FlairSequenceLabelingConfig:
    return FlairSequenceLabelingBasicConfig(hidden_size=256, max_epochs=1, mini_batch_size=256)


@pytest.fixture(scope="module")
def sequence_labeling_preprocessing_pipeline(
    result_path: "TemporaryDirectory[str]",
    ner_dataset_name: str,
) -> Tuple[PreprocessingPipeline[str, datasets.DatasetDict, Corpus], "TemporaryDirectory[str]"]:
    dataset = Dataset(ner_dataset_name)
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
    default_config: FlairSequenceLabelingConfig,
) -> Tuple[
    ModelEvaluationPipeline[str, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]],
    "TemporaryDirectory[str]",
]:
    pipeline = FlairSequenceLabelingEvaluationPipeline(
        dataset_path=result_path.name,
        embedding_name=embedding_name,
        output_path=result_path.name,
        config=default_config,
        persist_path=None,
    )
    return pipeline, result_path


def test_sequence_labeling_preprocessing_pipeline(
    result_path: "TemporaryDirectory[str]",
    ner_dataset_name: str,
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
        result["seqeval__mode_None__scheme_None"]["overall_accuracy"], 0.0024630
    )
    np.testing.assert_almost_equal(result["seqeval__mode_None__scheme_None"]["overall_f1"], 0)

    path.cleanup()
