from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union

import datasets
from flair.data import Corpus
from numpy import typing as nptyping

from embeddings.data.data_loader import HuggingFaceDataLoader
from embeddings.data.dataset import HuggingFaceDataset
from embeddings.data.io import T_path
from embeddings.embedding.auto_flair import AutoFlairDocumentPoolEmbedding, DocumentEmbedding
from embeddings.embedding.flair_embedding import FlairDocumentPoolEmbedding
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.flair_task.text_pair_classification import TextPairClassification
from embeddings.transformation.flair_transformation.pair_classification_corpus_transformation import (
    PairClassificationCorpusTransformation,
)
from embeddings.transformation.flair_transformation.split_sample_corpus_transformation import (
    SampleSplitsFlairCorpusTransformation,
)
from embeddings.transformation.transformation import Transformation
from embeddings.utils.json_dict_persister import JsonPersister


class FlairPairClassificationPipeline(
    StandardPipeline[
        str, datasets.DatasetDict, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]
    ]
):
    def __init__(
        self,
        embedding_name: str,
        dataset_name: str,
        input_columns_names_pair: Tuple[str, str],
        target_column_name: str,
        output_path: T_path,
        evaluation_filename: str = "evaluation.json",
        document_embedding_cls: Union[str, Type[DocumentEmbedding]] = FlairDocumentPoolEmbedding,
        sample_missing_splits: Optional[Tuple[Optional[float], Optional[float]]] = None,
        seed: int = 441,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
        load_model_kwargs: Optional[Dict[str, Any]] = None,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        output_path = Path(output_path)
        dataset = HuggingFaceDataset(
            dataset_name, **load_dataset_kwargs if load_dataset_kwargs else {}
        )
        data_loader = HuggingFaceDataLoader()
        transformation: Union[
            Transformation[datasets.DatasetDict, Corpus], Transformation[Corpus, Corpus]
        ]
        transformation = PairClassificationCorpusTransformation(
            input_columns_names_pair, target_column_name
        )
        if sample_missing_splits:
            transformation = transformation.then(
                SampleSplitsFlairCorpusTransformation(*sample_missing_splits, seed=seed)
            )
        embedding = AutoFlairDocumentPoolEmbedding.from_hub(
            repo_id=embedding_name,
            document_embedding_cls=document_embedding_cls,
            **load_model_kwargs if load_model_kwargs else {}
        )
        task = TextPairClassification(
            output_path,
            task_model_kwargs=task_model_kwargs,
            task_train_kwargs=task_train_kwargs,
        )
        model = FlairModel(embedding, task)
        evaluator = TextClassificationEvaluator().persisting(
            JsonPersister(path=output_path.joinpath(evaluation_filename))
        )
        super().__init__(dataset, data_loader, transformation, model, evaluator)
