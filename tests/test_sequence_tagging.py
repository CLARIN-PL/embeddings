from tempfile import TemporaryDirectory
from typing import Dict, List, Any, Tuple

import datasets
import numpy as np
import pytest
from embeddings.data.hugging_face_data_loader import HuggingFaceDataLoader
from embeddings.data.hugging_face_dataset import HuggingFaceDataset
from embeddings.embedding.flair_embedding import FlairTransformerWordEmbedding
from embeddings.evaluator.sequence_tagging_evaluator import SequenceTaggingEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.flair.sequence_tagging import SequenceTagging
from embeddings.transformation.flair.column_corpus_transformation import ColumnCorpusTransformation
from embeddings.transformation.flair.corpus_transformations import (
    DownsampleFlairCorpusTransformation,
)
from flair.data import Corpus


@pytest.fixture  # type: ignore
def sequence_tagging_pipeline(
    result_path: "TemporaryDirectory[str]",
) -> Tuple[
    StandardPipeline[str, datasets.DatasetDict, Corpus, Dict[str, np.ndarray], List[Any]],
    "TemporaryDirectory[str]",
]:
    dataset = HuggingFaceDataset("clarin-pl/nkjp-pos")
    data_loader = HuggingFaceDataLoader()
    transformation = ColumnCorpusTransformation("tokens", "pos_tags").then(
        DownsampleFlairCorpusTransformation(percentage=0.0025)
    )
    embedding = FlairTransformerWordEmbedding("allegro/herbert-base-cased")
    task = SequenceTagging(result_path.name, hidden_size=256, task_train_kwargs={"max_epochs": 1})
    model = FlairModel(embedding, task)
    evaluator = SequenceTaggingEvaluator()

    pipeline = StandardPipeline(dataset, data_loader, transformation, model, evaluator)
    return pipeline, result_path


def test_sequence_tagging_pipeline(
    sequence_tagging_pipeline: Tuple[
        StandardPipeline[str, datasets.DatasetDict, Corpus, Dict[str, np.ndarray], List[Any]],
        "TemporaryDirectory[str]",
    ]
) -> None:
    pipeline, path = sequence_tagging_pipeline
    result = pipeline.run()
    path.cleanup()

    assert 0 <= result[0]["overall_f1"] <= 1
