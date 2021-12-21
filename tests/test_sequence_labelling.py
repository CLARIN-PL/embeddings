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
from embeddings.data.dataset import HuggingFaceDataset
from embeddings.embedding.auto_flair import AutoFlairWordEmbedding
from embeddings.evaluator.sequence_labeling_evaluator import SequenceLabelingEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.flair_task.sequence_labeling import SequenceLabeling
from embeddings.transformation.flair_transformation.column_corpus_transformation import (
    ColumnCorpusTransformation,
)
from embeddings.transformation.flair_transformation.downsample_corpus_transformation import (
    DownsampleFlairCorpusTransformation,
)
from embeddings.transformation.flair_transformation.split_sample_corpus_transformation import (
    SampleSplitsFlairCorpusTransformation,
)


@pytest.fixture
def pos_tagging_pipeline(
    result_path: "TemporaryDirectory[str]",
) -> Tuple[
    StandardPipeline[
        str, datasets.DatasetDict, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]
    ],
    "TemporaryDirectory[str]",
]:
    dataset = HuggingFaceDataset("clarin-pl/nkjp-pos")
    data_loader = HuggingFaceDataLoader()
    transformation = ColumnCorpusTransformation("tokens", "pos_tags").then(
        DownsampleFlairCorpusTransformation(percentage=0.001)
    )
    embedding = AutoFlairWordEmbedding.from_hub("allegro/herbert-base-cased")
    task = SequenceLabeling(result_path.name, hidden_size=256, task_train_kwargs={"max_epochs": 1})
    model = FlairModel(embedding, task)
    evaluator = SequenceLabelingEvaluator(
        evaluation_mode=SequenceLabelingEvaluator.EvaluationMode.UNIT
    )

    pipeline = StandardPipeline(dataset, data_loader, transformation, model, evaluator)
    return pipeline, result_path


@pytest.fixture
def ner_tagging_pipeline(
    result_path: "TemporaryDirectory[str]",
) -> Tuple[
    StandardPipeline[
        str, datasets.DatasetDict, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]
    ],
    "TemporaryDirectory[str]",
]:
    dataset = HuggingFaceDataset("clarin-pl/kpwr-ner")
    data_loader = HuggingFaceDataLoader()
    transformation = (
        ColumnCorpusTransformation("tokens", "ner")
        .then(SampleSplitsFlairCorpusTransformation(dev_fraction=0.1, seed=441))
        .then(DownsampleFlairCorpusTransformation(percentage=0.005))
    )
    embedding = AutoFlairWordEmbedding.from_hub("allegro/herbert-base-cased")
    task = SequenceLabeling(
        result_path.name,
        hidden_size=256,
        task_train_kwargs={"max_epochs": 1, "mini_batch_size": 256},
    )
    model = FlairModel(embedding, task)
    evaluator = SequenceLabelingEvaluator()

    pipeline = StandardPipeline(dataset, data_loader, transformation, model, evaluator)
    return pipeline, result_path


def test_pos_tagging_pipeline(
    pos_tagging_pipeline: Tuple[
        StandardPipeline[
            str, datasets.DatasetDict, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]
        ],
        "TemporaryDirectory[str]",
    ],
) -> None:
    flair.set_seed(441)
    flair.device = torch.device("cpu")
    pipeline, path = pos_tagging_pipeline
    result = pipeline.run()
    path.cleanup()

    np.testing.assert_almost_equal(result["UnitSeqeval"]["overall_f1"], 0.1603053)


def test_ner_tagging_pipeline(
    ner_tagging_pipeline: Tuple[
        StandardPipeline[
            str, datasets.DatasetDict, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]
        ],
        "TemporaryDirectory[str]",
    ],
) -> None:
    flair.set_seed(441)
    flair.device = torch.device("cpu")
    pipeline, path = ner_tagging_pipeline
    result = pipeline.run()
    path.cleanup()

    np.testing.assert_almost_equal(result["seqeval__mode_None__scheme_None"]["overall_f1"], 0)
