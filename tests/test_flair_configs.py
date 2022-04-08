from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Tuple

import datasets
import flair
import pytest
import torch
from _pytest.tmpdir import TempdirFactory
from flair.data import Corpus
from numpy import typing as nptyping

from embeddings.config.flair_config import (
    FlairSequenceLabelingAdvancedConfig,
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
def result_path(tmpdir_factory: TempdirFactory) -> Path:
    path = tmpdir_factory.mktemp(__name__)
    return Path(path)


@pytest.fixture(scope="module")
def flair_basic_config() -> FlairSequenceLabelingConfig:
    return FlairSequenceLabelingBasicConfig(hidden_size=32, max_epochs=1, mini_batch_size=64)


@pytest.fixture(scope="module")
def flair_advanced_config() -> FlairSequenceLabelingConfig:
    return FlairSequenceLabelingAdvancedConfig(
        hidden_size=32,
        task_train_kwargs={"learning_rate": 1e-3, "mini_batch_size": 32, "max_epochs": 1},
        task_model_kwargs={
            "use_crf": True,
            "use_rnn": True,
            "rnn_type": "LSTM",
            "rnn_layers": 1,
            "dropout": 0.0,
            "word_dropout": 0.05,
            "locked_dropout": 0.5,
            "reproject_embeddings": True,
        },
        load_model_kwargs={},
    )


@pytest.fixture(scope="module")
def sequence_labeling_preprocessing_pipeline(
    result_path: Path,
) -> Tuple[PreprocessingPipeline[str, datasets.DatasetDict, Corpus], Path]:
    dataset = Dataset("clarin-pl/kpwr-ner")
    data_loader = HuggingFaceDataLoader()
    transformation = (
        ColumnCorpusTransformation("tokens", "ner")
        .then(SampleSplitsFlairCorpusTransformation(dev_fraction=0.1, test_fraction=0.1, seed=441))
        .then(DownsampleFlairCorpusTransformation(*(0.005, 0.005, 0.005)))
        .persisting(FlairConllPersister(str(result_path)))
    )
    pipeline = PreprocessingPipeline(
        dataset=dataset, data_loader=data_loader, transformation=transformation
    )
    return pipeline, result_path


@pytest.fixture(scope="module")
def sequence_labeling_evaluation_pipeline(
    result_path: Path,
    flair_basic_config: FlairSequenceLabelingConfig,
) -> Tuple[
    ModelEvaluationPipeline[str, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]],
    Path,
]:
    pipeline = FlairSequenceLabelingEvaluationPipeline(
        dataset_path=str(result_path),
        embedding_name="hf-internal-testing/tiny-albert",
        output_path=result_path,
        config=flair_basic_config,
        persist_path=None,
    )
    return pipeline, result_path


def test_sequence_labeling_basic_config(
    result_path: Path,
    sequence_labeling_preprocessing_pipeline: Tuple[
        PreprocessingPipeline[str, datasets.DatasetDict, Corpus], Path
    ],
    flair_basic_config: FlairSequenceLabelingConfig,
) -> None:
    flair.set_seed(441)
    flair.device = torch.device("cpu")

    preprocessing_pipeline, path = sequence_labeling_preprocessing_pipeline
    preprocessing_pipeline.run()

    pipeline = FlairSequenceLabelingEvaluationPipeline(
        dataset_path=str(result_path),
        embedding_name="hf-internal-testing/tiny-albert",
        output_path=result_path,
        config=flair_basic_config,
        persist_path=None,
    )
    pipeline.run()


def test_sequence_labeling_advanced_config(
    result_path: Path,
    sequence_labeling_preprocessing_pipeline: Tuple[
        PreprocessingPipeline[str, datasets.DatasetDict, Corpus], Path
    ],
    flair_advanced_config: FlairSequenceLabelingConfig,
) -> None:
    flair.set_seed(441)
    flair.device = torch.device("cpu")

    preprocessing_pipeline, path = sequence_labeling_preprocessing_pipeline
    preprocessing_pipeline.run()

    pipeline = FlairSequenceLabelingEvaluationPipeline(
        dataset_path=str(result_path),
        embedding_name="hf-internal-testing/tiny-albert",
        output_path=result_path,
        config=flair_advanced_config,
        persist_path=None,
    )
    pipeline.run()


def test_flair_advanced_config_from_basic() -> None:
    basic_config = FlairSequenceLabelingBasicConfig()
    config = FlairSequenceLabelingAdvancedConfig.from_basic()
    for field in fields(config):
        assert getattr(config, field.name) == getattr(basic_config, field.name)
