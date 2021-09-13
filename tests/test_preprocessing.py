import random
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict

import datasets
import flair
import numpy as np
import pytest
from flair.data import Corpus

from embeddings.data.data_loader import HuggingFaceDataLoader
from embeddings.data.dataset import HuggingFaceDataset
from embeddings.pipeline.evaluation_pipeline import (
    FlairSequenceLabellingEvaluationPipeline,
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

random.seed(141)


@pytest.fixture
def embedding_name() -> str:
    return "allegro/herbert-base-cased"


@pytest.fixture
def ner_dataset_name() -> str:
    return "clarin-pl/kpwr-ner"


@pytest.fixture
def ner_input_column_name() -> str:
    return "tokens"


@pytest.fixture
def ner_target_column_name() -> str:
    return "ner"


@pytest.fixture
def hidden_size() -> int:
    return 256


@pytest.fixture
def task_train_kwargs() -> Dict[str, int]:
    return {"max_epochs": 1, "mini_batch_size": 256}


@pytest.fixture
def sequence_labelling_preprocessing_pipeline(
    result_path: "TemporaryDirectory[str]",
    embedding_name: str,
    ner_dataset_name: str,
    ner_input_column_name: str,
    ner_target_column_name: str,
) -> PreprocessingPipeline[str, datasets.DatasetDict, Corpus]:
    dataset = HuggingFaceDataset(ner_dataset_name)
    data_loader = HuggingFaceDataLoader()
    transformation = (
        ColumnCorpusTransformation(ner_input_column_name, ner_target_column_name)
        .then(DownsampleFlairCorpusTransformation(percentage=0.005))
        .persisting(FlairConllPersister(result_path.name))
    )
    pipeline = PreprocessingPipeline(
        dataset=dataset, data_loader=data_loader, transformation=transformation
    )
    return pipeline


@pytest.fixture
def sequence_labelling_evaluation_pipeline(
    result_path: "TemporaryDirectory[str]",
    embedding_name: str,
    ner_dataset_name: str,
    hidden_size: int,
    task_train_kwargs: Dict[str, int],
) -> ModelEvaluationPipeline[str, Corpus, Dict[str, np.ndarray], Dict[str, Any]]:

    persist_out_path = Path(result_path.name, ner_dataset_name, f"{embedding_name}.json")
    persist_out_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(result_path.name, ner_dataset_name, embedding_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = FlairSequenceLabellingEvaluationPipeline(
        dataset_path=result_path.name,
        output_path=result_path.name,
        fine_tune_embeddings=False,
        hidden_size=hidden_size,
        embedding_name=embedding_name,
        persist_path=str(persist_out_path),
        task_train_kwargs=task_train_kwargs,
    )
    return pipeline


def test_sequence_labelling_preprocessing_pipeline(
    result_path: "TemporaryDirectory[str]",
    embedding_name: str,
    ner_dataset_name: str,
    hidden_size: int,
    task_train_kwargs: Dict[str, int],
    sequence_labelling_preprocessing_pipeline: StandardPipeline[
        str, datasets.DatasetDict, Corpus, Dict[str, np.ndarray], Dict[str, Any]
    ],
    sequence_labelling_evaluation_pipeline: ModelEvaluationPipeline[
        str, Corpus, Dict[str, np.ndarray], Dict[str, Any]
    ],
) -> None:
    flair.set_seed(441)

    preprocessing_pipeline = sequence_labelling_preprocessing_pipeline
    _ = preprocessing_pipeline.run()

    evaluation_pipeline = sequence_labelling_evaluation_pipeline
    result = evaluation_pipeline.run()

    np.testing.assert_almost_equal(
        result["seqeval__mode_None__scheme_None"]["overall_accuracy"], 0.28888888888888886
    )
    np.testing.assert_almost_equal(result["seqeval__mode_None__scheme_None"]["overall_f1"], 0)

    result_path.cleanup()
