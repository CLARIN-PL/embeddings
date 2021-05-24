from pathlib import Path
from pprint import pprint

import typer
from flair.embeddings import TransformerDocumentEmbeddings
import pytest
from embeddings.data.hugging_face_data_loader import HuggingFaceDataLoader
from embeddings.data.hugging_face_dataset import HuggingFaceDataset
from embeddings.embedding.flair_embedding import FlairTransformerDocumentEmbedding
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.flair_task import FlairTextClassification
from embeddings.transformation.transformation import Transformations
from embeddings.transformation.flair_transformations import (
    ToFlairCorpusTransformation,
    DownsampleFlairCorpusTransformation,
)
from experimental.defaults import RESULTS_PATH
import numpy as np


@pytest.fixture
def text_classification_pipeline() -> StandardPipeline:
    dataset = HuggingFaceDataset(
        "clarin-pl/polemo2-official",
        train_domains=["reviews"],
        dev_domains=["reviews"],
        test_domains=["reviews"],
        text_cfg="sentence",
    )
    data_loader = HuggingFaceDataLoader()
    transformation = Transformations(
        [
            ToFlairCorpusTransformation("text", "target"),
            DownsampleFlairCorpusTransformation(percentage=0.01),
        ]
    )
    embedding = FlairTransformerDocumentEmbedding("allegro/herbert-base-cased")
    task = FlairTextClassification(RESULTS_PATH, task_train_kwargs={"max_epochs": 1})
    model = FlairModel(embedding, task)
    evaluator = TextClassificationEvaluator()
    pipeline = StandardPipeline(dataset, data_loader, transformation, model, evaluator)
    return pipeline


def test_text_classification_pipeline(text_classification_pipeline: StandardPipeline) -> None:
    result = text_classification_pipeline.run()
    assert 0 < result[0]["accuracy"] < 1
