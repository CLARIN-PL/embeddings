from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import datasets
import flair
import numpy as np
import pytest
from flair.data import Corpus
from numpy import typing as nptyping

from embeddings.data.data_loader import HuggingFaceDataLoader
from embeddings.data.dataset import HuggingFaceDataset
from embeddings.embedding.auto_flair import AutoFlairDocumentEmbedding
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.flair_task.text_pair_classification import TextPairClassification
from embeddings.transformation.flair_transformation.downsample_corpus_transformation import (
    DownsampleFlairCorpusTransformation,
)
from embeddings.transformation.flair_transformation.pair_classification_corpus_transformation import (
    PairClassificationCorpusTransformation,
)
from embeddings.transformation.flair_transformation.split_sample_corpus_transformation import (
    SampleSplitsFlairCorpusTransformation,
)


@pytest.fixture(scope="module")
def text_pair_classification_pipeline(
    result_path: "TemporaryDirectory[str]",
) -> Tuple[
    StandardPipeline[
        str, datasets.DatasetDict, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]
    ],
    "TemporaryDirectory[str]",
]:
    dataset = HuggingFaceDataset("clarin-pl/cst-wikinews")
    data_loader = HuggingFaceDataLoader()
    transformation = (
        PairClassificationCorpusTransformation(("sentence_1", "sentence_2"), "label")
        .then(DownsampleFlairCorpusTransformation(percentage=0.1, stratify=True))
        .then(SampleSplitsFlairCorpusTransformation(dev_fraction=0.1, seed=441, stratify=True))
    )
    embedding = AutoFlairDocumentEmbedding.from_hub("allegro/herbert-base-cased")
    task = TextPairClassification(result_path.name, task_train_kwargs={"max_epochs": 1})
    model = FlairModel(embedding, task)
    evaluator = TextClassificationEvaluator()
    pipeline = StandardPipeline(dataset, data_loader, transformation, model, evaluator)
    return pipeline, result_path


def test_text_pair_classification_pipeline(
    text_pair_classification_pipeline: Tuple[
        StandardPipeline[
            str, datasets.DatasetDict, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]
        ],
        "TemporaryDirectory[str]",
    ],
) -> None:
    flair.set_seed(441)
    pipeline, path = text_pair_classification_pipeline
    result = pipeline.run()
    path.cleanup()
    np.testing.assert_almost_equal(result["accuracy"]["accuracy"], 0.1538461)
    np.testing.assert_almost_equal(result["f1__average_macro"]["f1"], 0.0222222)
    np.testing.assert_almost_equal(result["precision__average_macro"]["precision"], 0.0128205)
    np.testing.assert_almost_equal(result["recall__average_macro"]["recall"], 0.0833333)


@pytest.fixture(scope="module")
def text_pair_classification_pipeline_local_embedding(
    result_path: "TemporaryDirectory[str]",
    embedding_path: Path = Path("../wiki-forms-all-100-cbow-ns-30-it100.txt.gz"),
    model_type_reference: str = "embeddings.embedding.static.word2vec.IPIPANWord2VecEmbedding",
) -> Tuple[
    StandardPipeline[
        str, datasets.DatasetDict, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]
    ],
    "TemporaryDirectory[str]",
]:
    dataset = HuggingFaceDataset("clarin-pl/cst-wikinews")
    data_loader = HuggingFaceDataLoader()
    transformation = (
        PairClassificationCorpusTransformation(("sentence_1", "sentence_2"), "label")
        .then(DownsampleFlairCorpusTransformation(percentage=0.1, stratify=True))
        .then(SampleSplitsFlairCorpusTransformation(dev_fraction=0.1, seed=441, stratify=True))
    )
    embedding = AutoFlairDocumentEmbedding.from_file(embedding_path, model_type_reference)
    task = TextPairClassification(result_path.name, task_train_kwargs={"max_epochs": 1})
    model = FlairModel(embedding, task)
    evaluator = TextClassificationEvaluator()
    pipeline = StandardPipeline(dataset, data_loader, transformation, model, evaluator)
    return pipeline, result_path


def test_text_pair_classification_pipeline_local_embedding(
    text_pair_classification_pipeline_local_embedding: Tuple[
        StandardPipeline[
            str, datasets.DatasetDict, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]
        ],
        "TemporaryDirectory[str]",
    ],
) -> None:
    flair.set_seed(441)
    pipeline, path = text_pair_classification_pipeline_local_embedding
    result = pipeline.run()
    path.cleanup()
    np.testing.assert_almost_equal(result["accuracy"]["accuracy"], 0.1025641)
    np.testing.assert_almost_equal(result["f1__average_macro"]["f1"], 0.062160)
    np.testing.assert_almost_equal(result["precision__average_macro"]["precision"], 0.051282)
    np.testing.assert_almost_equal(result["recall__average_macro"]["recall"], 0.08205128)
