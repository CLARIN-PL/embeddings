from pprint import pprint
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import datasets
import flair
import numpy as np
import pytest
from flair.data import Corpus

from embeddings.data.hugging_face_data_loader import HuggingFaceDataLoader
from embeddings.data.hugging_face_dataset import HuggingFaceDataset
from embeddings.embedding.flair_embedding import FlairTransformerDocumentEmbedding
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


@pytest.fixture  # type: ignore
def text_pair_classification_pipeline(
    result_path: "TemporaryDirectory[str]",
) -> Tuple[
    StandardPipeline[str, datasets.DatasetDict, Corpus, Dict[str, np.ndarray], Dict[str, Any]],
    "TemporaryDirectory[str]",
]:
    dataset = HuggingFaceDataset("clarin-pl/cst-wikinews")
    data_loader = HuggingFaceDataLoader()
    transformation = PairClassificationCorpusTransformation(
        ("sentence_1", "sentence_2"), "label"
    ).then(DownsampleFlairCorpusTransformation(percentage=0.1))
    embedding = FlairTransformerDocumentEmbedding("allegro/herbert-base-cased")
    task = TextPairClassification(result_path.name, task_train_kwargs={"max_epochs": 1})
    model = FlairModel(embedding, task)
    evaluator = TextClassificationEvaluator()
    pipeline = StandardPipeline(dataset, data_loader, transformation, model, evaluator)
    return pipeline, result_path


def test_text_pair_classification_pipeline(
    text_pair_classification_pipeline: Tuple[
        StandardPipeline[str, datasets.DatasetDict, Corpus, Dict[str, np.ndarray], Dict[str, Any]],
        "TemporaryDirectory[str]",
    ],
) -> None:
    flair.set_seed(441)
    pipeline, path = text_pair_classification_pipeline
    result = pipeline.run()
    path.cleanup()
    pprint(result)
    np.testing.assert_almost_equal(result["accuracy"]["accuracy"], 0.0789473)
    np.testing.assert_almost_equal(result["f1__average_macro"]["f1"], 0.0547785)
    np.testing.assert_almost_equal(result["precision__average_macro"]["precision"], 0.0958230)
    np.testing.assert_almost_equal(result["recall__average_macro"]["recall"], 0.1212121)
