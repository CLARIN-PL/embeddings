from typing import Any, Dict, Optional, Tuple

import datasets
import numpy as np
from flair.data import Corpus

from embeddings.data.hugging_face_data_loader import HuggingFaceDataLoader
from embeddings.data.hugging_face_dataset import HuggingFaceDataset
from embeddings.data.io import T_path
from embeddings.embedding.flair_embedding import FlairTransformerDocumentEmbedding
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.flair_task.text_pair_classification import TextPairClassification
from embeddings.transformation.flair_transformation.pair_classification_corpus_transformation import (
    PairClassificationCorpusTransformation,
)


class HuggingFacePairClassificationPipeline(
    StandardPipeline[str, datasets.DatasetDict, Corpus, Dict[str, np.ndarray], Dict[str, Any]]
):
    def __init__(
        self,
        embedding_name: str,
        dataset_name: str,
        input_columns_names_pair: Tuple[str, str],
        target_column_name: str,
        output_path: T_path,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
    ):
        dataset = HuggingFaceDataset(dataset_name)
        data_loader = HuggingFaceDataLoader()
        transformation = PairClassificationCorpusTransformation(
            input_columns_names_pair, target_column_name
        )
        embedding = FlairTransformerDocumentEmbedding(embedding_name)
        task = TextPairClassification(
            output_path,
            task_model_kwargs=task_model_kwargs,
            task_train_kwargs=task_train_kwargs,
        )
        model = FlairModel(embedding, task)
        evaluator = TextClassificationEvaluator()
        super().__init__(dataset, data_loader, transformation, model, evaluator)
