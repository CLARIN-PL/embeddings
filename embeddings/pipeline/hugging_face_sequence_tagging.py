from typing import Any, Dict

import datasets
import numpy as np
from flair.data import Corpus

from embeddings.data.hugging_face_data_loader import HuggingFaceDataLoader
from embeddings.data.hugging_face_dataset import HuggingFaceDataset
from embeddings.data.io import T_path
from embeddings.embedding.flair_embedding import FlairTransformerWordEmbedding
from embeddings.evaluator.sequence_tagging_evaluator import POSTaggingEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.flair_task.sequence_tagging import SequenceTagging
from embeddings.transformation.flair_transformation.column_corpus_transformation import (
    ColumnCorpusTransformation,
)


class HuggingFaceSequenceTaggingPipeline(
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
    ):
        dataset = HuggingFaceDataset(dataset_name)
        data_loader = HuggingFaceDataLoader()
        transformation = ColumnCorpusTransformation(input_column_name, target_column_name)
        embedding = FlairTransformerWordEmbedding(embedding_name)
        task = SequenceTagging(output_path, hidden_size=hidden_size)
        model = FlairModel(embedding, task)
        # todo: should be parametrised?
        evaluator = POSTaggingEvaluator()
        super().__init__(dataset, data_loader, transformation, model, evaluator)
