from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import datasets
import flair
import numpy as np
import pytest
from flair.data import Corpus

from embeddings.data.data_loader import HuggingFaceDataLoader
from embeddings.data.dataset import HuggingFaceDataset
from embeddings.embedding.auto_flair import AutoFlairDocumentEmbedding
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.flair_task.text_classification import TextClassification
from embeddings.transformation.flair_transformation.classification_corpus_transformation import (
    ClassificationCorpusTransformation,
)
from embeddings.transformation.flair_transformation.downsample_corpus_transformation import (
    DownsampleFlairCorpusTransformation,
)


@pytest.fixture
def text_classification_pipeline(
    result_path: "TemporaryDirectory[str]",
) -> Tuple[
    StandardPipeline[str, datasets.DatasetDict, Corpus, Dict[str, np.ndarray], Dict[str, Any]],
    "TemporaryDirectory[str]",
]:
    dataset = HuggingFaceDataset(
        "clarin-pl/polemo2-official",
        train_domains=["reviews"],
        dev_domains=["reviews"],
        test_domains=["reviews"],
        text_cfg="sentence",
    )
    data_loader = HuggingFaceDataLoader()
    transformation = ClassificationCorpusTransformation("text", "target").then(
        DownsampleFlairCorpusTransformation(percentage=0.01, stratify=False)
    )
    embedding = AutoFlairDocumentEmbedding.from_hub("allegro/herbert-base-cased")
    task = TextClassification(result_path.name, task_train_kwargs={"max_epochs": 1})
    model = FlairModel(embedding, task)
    evaluator = TextClassificationEvaluator()
    pipeline = StandardPipeline(dataset, data_loader, transformation, model, evaluator)
    return pipeline, result_path


def test_text_classification_pipeline(
    text_classification_pipeline: Tuple[
        StandardPipeline[str, datasets.DatasetDict, Corpus, Dict[str, np.ndarray], Dict[str, Any]],
        "TemporaryDirectory[str]",
    ],
) -> None:
    flair.set_seed(441)
    pipeline, path = text_classification_pipeline
    result = pipeline.run()
    path.cleanup()
    np.testing.assert_almost_equal(result["accuracy"]["accuracy"], 0.3333333)
    np.testing.assert_almost_equal(result["f1__average_macro"]["f1"], 0.1666666)
    np.testing.assert_almost_equal(result["precision__average_macro"]["precision"], 0.1111111)
    np.testing.assert_almost_equal(result["recall__average_macro"]["recall"], 0.3333333)
