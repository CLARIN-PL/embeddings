from typing import Dict, List, Any
import datasets
import numpy as np
import pytest
from embeddings.data.hugging_face_data_loader import HuggingFaceDataLoader
from embeddings.data.hugging_face_dataset import HuggingFaceDataset
from embeddings.embedding.flair_embedding import FlairTransformerDocumentEmbedding
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.flair.text_classification import TextClassification
from embeddings.transformation.flair.corpus_transformations import DownsampleFlairCorpusTransformation
from embeddings.transformation.flair.classification_corpus_transformation import ClassificationCorpusTransformation

from embeddings.transformation.transformation import Transformations
from experimental.defaults import RESULTS_PATH
from flair.data import Corpus


@pytest.fixture  # type: ignore
def text_classification_pipeline() -> StandardPipeline[
    str, datasets.DatasetDict, Corpus, Dict[str, np.ndarray], List[Any]
]:
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
            ClassificationCorpusTransformation("text", "target"),
            DownsampleFlairCorpusTransformation(percentage=0.01),
        ]
    )
    embedding = FlairTransformerDocumentEmbedding("allegro/herbert-base-cased")
    task = TextClassification(RESULTS_PATH, task_train_kwargs={"max_epochs": 1})
    model = FlairModel(embedding, task)
    evaluator = TextClassificationEvaluator()
    pipeline = StandardPipeline(dataset, data_loader, transformation, model, evaluator)
    return pipeline


def test_text_classification_pipeline(
    text_classification_pipeline: StandardPipeline[
        str, datasets.DatasetDict, Corpus, Dict[str, np.ndarray], List[Any]
    ]
) -> None:
    result = text_classification_pipeline.run()
    assert 0 <= result[1]["f1"] <= 1
