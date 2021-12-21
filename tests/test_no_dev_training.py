from tempfile import TemporaryDirectory
from typing import Any, Dict

import datasets
import flair
import numpy as np
import pytest
import torch
from flair.data import Corpus

from embeddings.data.data_loader import HuggingFaceDataLoader
from embeddings.data.dataset import HuggingFaceDataset
from embeddings.pipeline.evaluation_pipeline import (
    FlairSequenceLabelingEvaluationPipeline,
    ModelEvaluationPipeline,
)
from embeddings.pipeline.preprocessing_pipeline import PreprocessingPipeline
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.transformation.flair_transformation.column_corpus_transformation import (
    ColumnCorpusTransformation,
)
from embeddings.transformation.flair_transformation.downsample_corpus_transformation import (
    DownsampleFlairCorpusTransformation,
)
from embeddings.utils.flair_corpus_persister import FlairConllPersister


@pytest.fixture
def result_path() -> "TemporaryDirectory[str]":
    return TemporaryDirectory()


@pytest.fixture
def embedding_name() -> str:
    return "allegro/herbert-base-cased"


@pytest.fixture
def ner_dataset_name() -> str:
    return "clarin-pl/kpwr-ner"


@pytest.fixture
def default_hidden_size() -> int:
    return 256


@pytest.fixture
def sequence_labeling_preprocessing_pipeline(
    result_path: "TemporaryDirectory[str]",
    embedding_name: str,
    ner_dataset_name: str,
) -> PreprocessingPipeline[str, datasets.DatasetDict, Corpus]:
    dataset = HuggingFaceDataset(ner_dataset_name)
    data_loader = HuggingFaceDataLoader()
    transformation = (
        ColumnCorpusTransformation("tokens", "ner")
        .then(DownsampleFlairCorpusTransformation(percentage=0.005, seed=14))
        .persisting(FlairConllPersister(result_path.name))
    )
    pipeline = PreprocessingPipeline(
        dataset=dataset, data_loader=data_loader, transformation=transformation
    )
    return pipeline


@pytest.fixture
def sequence_labeling_evaluation_pipeline(
    result_path: "TemporaryDirectory[str]",
    embedding_name: str,
    ner_dataset_name: str,
    default_hidden_size: int,
) -> ModelEvaluationPipeline[str, Corpus, Dict[str, np.ndarray], Dict[str, Any]]:

    pipeline = FlairSequenceLabelingEvaluationPipeline(
        dataset_path=result_path.name,
        embedding_name=embedding_name,
        output_path=result_path.name,
        hidden_size=default_hidden_size,
        persist_path=None,
        task_train_kwargs={"max_epochs": 1, "mini_batch_size": 32},
    )
    return pipeline


def test_no_dev_pipeline(
    result_path: "TemporaryDirectory[str]",
    embedding_name: str,
    ner_dataset_name: str,
    sequence_labeling_preprocessing_pipeline: StandardPipeline[
        str, datasets.DatasetDict, Corpus, Dict[str, np.ndarray], Dict[str, Any]
    ],
    sequence_labeling_evaluation_pipeline: ModelEvaluationPipeline[
        str, Corpus, Dict[str, np.ndarray], Dict[str, Any]
    ],
) -> None:
    flair.set_seed(441)
    flair.device = torch.device("cpu")

    data: Corpus = sequence_labeling_preprocessing_pipeline.run()
    assert data.dev is None

    result = sequence_labeling_evaluation_pipeline.run()

    np.testing.assert_almost_equal(
        result["seqeval__mode_None__scheme_None"]["overall_accuracy"], 0.9
    )
    np.testing.assert_almost_equal(result["seqeval__mode_None__scheme_None"]["overall_f1"], 0)

    result_path.cleanup()
