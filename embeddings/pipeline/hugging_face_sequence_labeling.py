from typing import Any, Dict, Optional

import datasets
import numpy as np
from flair.data import Corpus

from embeddings.data.hugging_face_data_loader import HuggingFaceDataLoader
from embeddings.data.hugging_face_dataset import HuggingFaceDataset
from embeddings.data.io import T_path
from embeddings.embedding.flair_embedding import FlairTransformerWordEmbedding
from embeddings.evaluator.sequence_labeling_evaluator import SequenceLabelingEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.flair_task.sequence_labeling import SequenceLabeling
from embeddings.transformation.flair_transformation.column_corpus_transformation import (
    ColumnCorpusTransformation,
)


class HuggingFaceSequenceLabelingPipeline(
    StandardPipeline[str, datasets.DatasetDict, Corpus, Dict[str, np.ndarray], Dict[str, Any]]
):
    def __init__(
        self,
        embedding_name: str,
        dataset_name: str,
        input_column_name: str,
        target_column_name: str,
        output_path: T_path,
        hidden_size: int,
        evaluation_mode: str = "conll",
        tagging_scheme: Optional[str] = None,
    ):
        dataset = HuggingFaceDataset(dataset_name)
        data_loader = HuggingFaceDataLoader()
        transformation = ColumnCorpusTransformation(input_column_name, target_column_name)
        embedding = FlairTransformerWordEmbedding(embedding_name)
        task = SequenceLabeling(output_path, hidden_size=hidden_size)
        model = FlairModel(embedding, task)
        evaluator = SequenceLabelingEvaluator(
            evaluation_mode=evaluation_mode, tagging_scheme=tagging_scheme
        )
        super().__init__(dataset, data_loader, transformation, model, evaluator)
