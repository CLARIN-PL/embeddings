from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import datasets
import flair
import numpy as np
import pytest
from flair.data import Corpus

from embeddings.data.hugging_face_data_loader import HuggingFaceDataLoader
from embeddings.data.hugging_face_dataset import HuggingFaceDataset
from embeddings.embedding.flair_embedding import FlairTransformerWordEmbedding
from embeddings.evaluator.sequence_tagging_evaluator import POSTaggingEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.flair_task.sequence_tagging import SequenceTagging
from embeddings.transformation.flair_transformation.column_corpus_transformation import (
    ColumnCorpusTransformation,
)
from embeddings.transformation.flair_transformation.downsample_corpus_transformation import (
    DownsampleFlairCorpusTransformation,
)


@pytest.fixture  # type: ignore
def sequence_tagging_pipeline(
    result_path: "TemporaryDirectory[str]",
) -> Tuple[
    StandardPipeline[str, datasets.DatasetDict, Corpus, Dict[str, np.ndarray], Dict[str, Any]],
    "TemporaryDirectory[str]",
]:
    dataset = HuggingFaceDataset("clarin-pl/nkjp-pos")
    data_loader = HuggingFaceDataLoader()
    transformation = ColumnCorpusTransformation("tokens", "pos_tags").then(
        DownsampleFlairCorpusTransformation(percentage=0.001)
    )
    embedding = FlairTransformerWordEmbedding("allegro/herbert-base-cased")
    task = SequenceTagging(result_path.name, hidden_size=256, task_train_kwargs={"max_epochs": 1})
    model = FlairModel(embedding, task)
    evaluator = POSTaggingEvaluator()

    pipeline = StandardPipeline(dataset, data_loader, transformation, model, evaluator)
    return pipeline, result_path


def test_sequence_tagging_pipeline(
    sequence_tagging_pipeline: Tuple[
        StandardPipeline[str, datasets.DatasetDict, Corpus, Dict[str, np.ndarray], Dict[str, Any]],
        "TemporaryDirectory[str]",
    ],
) -> None:
    flair.set_seed(441)
    pipeline, path = sequence_tagging_pipeline
    result = pipeline.run()
    path.cleanup()

    np.testing.assert_almost_equal(
        result["POSTaggingSeqevalMetric"]["overall_f1"], 0.14788732394366197
    )
