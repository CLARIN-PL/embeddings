from typing import Any, Dict, Generic, Optional, Type, Union

from flair.data import Corpus
from numpy import typing as nptyping
from typing_extensions import Literal

from embeddings.data.data_loader import (
    ConllFlairCorpusDataLoader,
    DataLoader,
    PickleFlairCorpusDataLoader,
)
from embeddings.data.dataset import BaseDataset, Data, Dataset
from embeddings.data.io import T_path
from embeddings.embedding.auto_flair import DocumentEmbedding
from embeddings.embedding.flair_embedding import FlairDocumentPoolEmbedding
from embeddings.embedding.flair_loader import (
    FlairDocumentPoolEmbeddingLoader,
    FlairWordEmbeddingLoader,
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
        dataset: BaseDataset[Data],
        data_loader: DataLoader[Data, LoaderResult],
        model: Model[LoaderResult, ModelResult],
        evaluator: Evaluator[ModelResult, EvaluationResult],
    ) -> None:
        self.dataset = dataset
        self.data_loader = data_loader
        self.model = model
        self.evaluator = evaluator

    def run(self, **kwargs: Any) -> EvaluationResult:
        loaded_data = self.data_loader.load(self.dataset)
        model_result = self.model.execute(loaded_data)
        return self.evaluator.evaluate(model_result)


class FlairTextClassificationEvaluationPipeline(
    ModelEvaluationPipeline[str, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]]
):
    def __init__(
        self,
        dataset_path: str,
        embedding_name: str,
        output_path: T_path,
        document_embedding_cls: Union[str, Type[DocumentEmbedding]] = FlairDocumentPoolEmbedding,
        model_type_reference: str = "",
        persist_path: Optional[T_path] = None,
        predict_subset: Literal["dev", "test"] = "test",
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
        load_model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        load_model_kwargs = {} if load_model_kwargs is None else load_model_kwargs
        dataset = Dataset(dataset=dataset_path)
        data_loader = PickleFlairCorpusDataLoader()

        embedding_loader = FlairDocumentPoolEmbeddingLoader(embedding_name, model_type_reference)
        embedding = embedding_loader.get_embedding(document_embedding_cls, **load_model_kwargs)

        task = TextClassification(
            output_path=output_path,
            task_train_kwargs=task_train_kwargs,
            task_model_kwargs=task_model_kwargs,
        )
        model = FlairModel(embedding=embedding, task=task, predict_subset=predict_subset)
        evaluator: Evaluator[Dict[str, Any], Dict[str, Any]] = TextClassificationEvaluator()
        if persist_path is not None:
            evaluator = evaluator.persisting(JsonPersister(path=persist_path))
        super().__init__(dataset, data_loader, model, evaluator)


class FlairTextPairClassificationEvaluationPipeline(
    ModelEvaluationPipeline[str, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]]
):
    def __init__(
        self,
        dataset_path: str,
        embedding_name: T_path,
        output_path: T_path,
        document_embedding_cls: Union[str, Type[DocumentEmbedding]] = FlairDocumentPoolEmbedding,
        model_type_reference: str = "",
        persist_path: Optional[T_path] = None,
        predict_subset: Literal["dev", "test"] = "test",
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
        load_model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        load_model_kwargs = {} if load_model_kwargs is None else load_model_kwargs
        dataset = Dataset(dataset=dataset_path)
        data_loader = PickleFlairCorpusDataLoader()

        embedding_loader = FlairDocumentPoolEmbeddingLoader(embedding_name, model_type_reference)
        embedding = embedding_loader.get_embedding(document_embedding_cls, **load_model_kwargs)

        task = TextPairClassification(
            output_path=output_path,
            task_train_kwargs=task_train_kwargs,
            task_model_kwargs=task_model_kwargs,
        )
        model = FlairModel(embedding=embedding, task=task, predict_subset=predict_subset)
        evaluator: Evaluator[Dict[str, Any], Dict[str, Any]] = TextClassificationEvaluator()
        if persist_path:
            evaluator = evaluator.persisting(JsonPersister(path=persist_path))
        super().__init__(dataset, data_loader, model, evaluator)


class FlairSequenceLabelingEvaluationPipeline(
    ModelEvaluationPipeline[str, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]]
):
    DEFAULT_EVAL_MODE = SequenceLabelingEvaluator.EvaluationMode.CONLL

    def __init__(
        self,
        dataset_path: str,
        embedding_name: T_path,
        output_path: T_path,
        hidden_size: int,
        model_type_reference: str = "",
        evaluation_mode: SequenceLabelingEvaluator.EvaluationMode = DEFAULT_EVAL_MODE,
        tagging_scheme: Optional[SequenceLabelingEvaluator.TaggingScheme] = None,
        persist_path: Optional[T_path] = None,
        predict_subset: Literal["dev", "test"] = "test",
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
        word_embedding_kwargs: Optional[Dict[str, Any]] = None,
    ):
        dataset = Dataset(dataset=dataset_path)
        data_loader = ConllFlairCorpusDataLoader()

        embedding_loader = FlairWordEmbeddingLoader(embedding_name, model_type_reference)
        embedding = embedding_loader.get_embedding(**word_embedding_kwargs or {})

        task = SequenceLabeling(
            output_path=output_path,
            hidden_size=hidden_size,
            task_train_kwargs=task_train_kwargs,
            task_model_kwargs=task_model_kwargs,
        )
        model = FlairModel(embedding=embedding, task=task, predict_subset=predict_subset)
        evaluator: Evaluator[Dict[str, Any], Dict[str, Any]] = SequenceLabelingEvaluator(
            evaluation_mode=evaluation_mode, tagging_scheme=tagging_scheme
        )
        if persist_path:
            evaluator = evaluator.persisting(JsonPersister(path=persist_path))
        super().__init__(dataset, data_loader, model, evaluator)
