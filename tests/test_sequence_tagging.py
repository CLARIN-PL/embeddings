from typing import Dict, List, Any

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
from embeddings.transformation.transformation import Transformations
from experimental.defaults import RESULTS_PATH
from flair.data import Corpus


@pytest.fixture  # type: ignore
def sequence_tagging_pipeline() -> StandardPipeline[
    str, datasets.DatasetDict, Corpus, Dict[str, np.ndarray], List[Any]
]:
    dataset = HuggingFaceDataset("clarin-pl/nkjp-pos")
    data_loader = HuggingFaceDataLoader()
    transformation = Transformations(
        [
            ColumnCorpusTransformation("tokens", "pos_tags"),
            DownsampleFlairCorpusTransformation(percentage=0.0025),
        ]
    )
    embedding = FlairTransformerWordEmbedding("allegro/herbert-base-cased")
    task = SequenceTagging(RESULTS_PATH, hidden_size=256, task_train_kwargs={"max_epochs": 1})
    model = FlairModel(embedding, task)
    evaluator = SequenceTaggingEvaluator()

    pipeline = StandardPipeline(dataset, data_loader, transformation, model, evaluator)
    return pipeline


def test_sequence_tagging_pipeline(
    sequence_tagging_pipeline: StandardPipeline[
        str, datasets.DatasetDict, Corpus, Dict[str, np.ndarray], List[Any]
    ]
) -> None:
    result = sequence_tagging_pipeline.run()
    assert 0 <= result[0]["overall_f1"] <= 1
