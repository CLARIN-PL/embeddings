from typing import Any, Dict, Generic, Optional

import numpy as np
from flair.data import Corpus

from embeddings.data.data_loader import (
    ConllFlairCorpusDataLoader,
    DataLoader,
    PickleFlairCorpusDataLoader,
)
from embeddings.data.dataset import Data, Dataset, LocalDataset
from embeddings.embedding.flair_embedding import (
    FlairTransformerDocumentEmbedding,
    FlairTransformerWordEmbedding,
)
from embeddings.evaluator.evaluator import Evaluator
from embeddings.evaluator.sequence_labeling_evaluator import SequenceLabelingEvaluator
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.model.model import Model
from embeddings.pipeline.pipeline import Pipeline
from embeddings.pipeline.standard_pipeline import EvaluationResult, LoaderResult, ModelResult
from embeddings.task.flair_task.sequence_labeling import SequenceLabeling
from embeddings.task.flair_task.text_classification import TextClassification
from embeddings.task.flair_task.text_pair_classification import TextPairClassification
from embeddings.utils.json_dict_persister import JsonPersister


class ModelEvaluationPipeline(
    Pipeline[EvaluationResult],
    Generic[Data, LoaderResult, ModelResult, EvaluationResult],
):
    def __init__(
        self,
        dataset: Dataset[Data],
        data_loader: DataLoader[Data, LoaderResult],
        model: Model[LoaderResult, ModelResult],
        evaluator: Evaluator[ModelResult, EvaluationResult],
    ) -> None:
        self.dataset = dataset
        self.data_loader = data_loader
        self.model = model
        self.evaluator = evaluator

    def run(self) -> EvaluationResult:
        loaded_data = self.data_loader.load(self.dataset)
        model_result = self.model.execute(loaded_data)
        return self.evaluator.evaluate(model_result)


class FlairTextClassificationEvaluationPipeline(
    ModelEvaluationPipeline[str, Corpus, Dict[str, np.ndarray], Dict[str, Any]]
):
    def __init__(
        self,
        dataset_path: str,
        embedding_name: str,
        fine_tune_embeddings: bool,
        output_path: str,
        persist_path: Optional[str] = None,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
    ):
        dataset = LocalDataset(dataset_path)
        data_loader = PickleFlairCorpusDataLoader()
        embedding = FlairTransformerDocumentEmbedding(
            name=embedding_name, fine_tune=fine_tune_embeddings
        )
        task = TextClassification(
            output_path=output_path,
            task_train_kwargs=task_train_kwargs,
            task_model_kwargs=task_model_kwargs,
        )
        model = FlairModel(embedding=embedding, task=task)
        evaluator: Evaluator[Dict[str, Any], Dict[str, Any]] = TextClassificationEvaluator()
        if persist_path is not None:
            evaluator = evaluator.persisting(JsonPersister(persist_path))
        super().__init__(dataset, data_loader, model, evaluator)


class FlairTextPairClassificationEvaluationPipeline(
    ModelEvaluationPipeline[str, Corpus, Dict[str, np.ndarray], Dict[str, Any]]
):
    def __init__(
        self,
        dataset_path: str,
        embedding_name: str,
        fine_tune_embeddings: bool,
        output_path: str,
        persist_path: Optional[str] = None,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
    ):
        dataset = LocalDataset(dataset_path)
        data_loader = PickleFlairCorpusDataLoader()
        embedding = FlairTransformerDocumentEmbedding(
            name=embedding_name, fine_tune=fine_tune_embeddings
        )
        task = TextPairClassification(
            output_path=output_path,
            task_train_kwargs=task_train_kwargs,
            task_model_kwargs=task_model_kwargs,
        )
        model = FlairModel(embedding=embedding, task=task)
        evaluator: Evaluator[Dict[str, Any], Dict[str, Any]] = TextClassificationEvaluator()
        if persist_path:
            evaluator = evaluator.persisting(JsonPersister(persist_path))
        super().__init__(dataset, data_loader, model, evaluator)


class FlairSequenceLabelingEvaluationPipeline(
    ModelEvaluationPipeline[str, Corpus, Dict[str, np.ndarray], Dict[str, Any]]
):
    def __init__(
        self,
        dataset_path: str,
        embedding_name: str,
        fine_tune_embeddings: bool,
        output_path: str,
        hidden_size: int,
        evaluation_mode: str = "conll",
        tagging_scheme: Optional[str] = None,
        persist_path: Optional[str] = None,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
    ):
        dataset = LocalDataset(dataset_path)
        data_loader = ConllFlairCorpusDataLoader()
        embedding = FlairTransformerWordEmbedding(
            name=embedding_name, fine_tune=fine_tune_embeddings
        )
        task = SequenceLabeling(
            output_path=output_path,
            hidden_size=hidden_size,
            task_train_kwargs=task_train_kwargs,
            task_model_kwargs=task_model_kwargs,
        )
        model = FlairModel(embedding=embedding, task=task)
        evaluator: Evaluator[Dict[str, Any], Dict[str, Any]] = SequenceLabelingEvaluator(
            evaluation_mode=evaluation_mode, tagging_scheme=tagging_scheme
        )
        if persist_path:
            evaluator = evaluator.persisting(JsonPersister(persist_path))
        super().__init__(dataset, data_loader, model, evaluator)
