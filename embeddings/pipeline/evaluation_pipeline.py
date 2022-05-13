from typing import Any, Generic, Optional

from flair.data import Corpus
from typing_extensions import Literal

from embeddings.config.flair_config import (
    FlairSequenceLabelingBasicConfig,
    FlairSequenceLabelingConfig,
    FlairTextClassificationBasicConfig,
    FlairTextClassificationConfig,
)
from embeddings.data.data_loader import (
    ConllFlairCorpusDataLoader,
    DataLoader,
    PickleFlairCorpusDataLoader,
)
from embeddings.data.dataset import BaseDataset, Data, Dataset
from embeddings.data.io import T_path
from embeddings.embedding.flair_loader import (
    FlairDocumentPoolEmbeddingLoader,
    FlairWordEmbeddingLoader,
)
from embeddings.evaluator.evaluation_results import (
    Predictions,
    SequenceLabelingEvaluationResults,
    TextClassificationEvaluationResults,
)
from embeddings.evaluator.evaluator import Evaluator
from embeddings.evaluator.sequence_labeling_evaluator import SequenceLabelingEvaluator
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.metric.sequence_labeling import EvaluationMode, TaggingScheme
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
    ModelEvaluationPipeline[str, Corpus, Predictions, TextClassificationEvaluationResults]
):
    def __init__(
        self,
        dataset_path: str,
        embedding_name: str,
        output_path: T_path,
        model_type_reference: str = "",
        config: FlairTextClassificationConfig = FlairTextClassificationBasicConfig(),
        persist_path: Optional[T_path] = None,
        predict_subset: Literal["dev", "test"] = "test",
    ):
        dataset = Dataset(dataset=dataset_path)
        data_loader = PickleFlairCorpusDataLoader()

        embedding_loader = FlairDocumentPoolEmbeddingLoader(embedding_name, model_type_reference)
        embedding = embedding_loader.get_embedding(
            config.document_embedding_cls, **config.load_model_kwargs
        )

        task = TextClassification(
            output_path,
            task_model_kwargs=config.task_model_kwargs,
            task_train_kwargs=config.task_train_kwargs,
        )
        model = FlairModel(embedding=embedding, task=task, predict_subset=predict_subset)
        if persist_path is not None:
            evaluator = TextClassificationEvaluator().persisting(JsonPersister(path=persist_path))
        else:
            evaluator = TextClassificationEvaluator()

        super().__init__(dataset, data_loader, model, evaluator)


class FlairTextPairClassificationEvaluationPipeline(
    ModelEvaluationPipeline[str, Corpus, Predictions, TextClassificationEvaluationResults]
):
    def __init__(
        self,
        dataset_path: str,
        embedding_name: T_path,
        output_path: T_path,
        model_type_reference: str = "",
        config: FlairTextClassificationConfig = FlairTextClassificationBasicConfig(),
        persist_path: Optional[T_path] = None,
        predict_subset: Literal["dev", "test"] = "test",
    ):
        dataset = Dataset(dataset=dataset_path)
        data_loader = PickleFlairCorpusDataLoader()

        embedding_loader = FlairDocumentPoolEmbeddingLoader(embedding_name, model_type_reference)
        embedding = embedding_loader.get_embedding(
            config.document_embedding_cls, **config.load_model_kwargs
        )

        task = TextPairClassification(
            output_path,
            task_model_kwargs=config.task_model_kwargs,
            task_train_kwargs=config.task_train_kwargs,
        )
        model = FlairModel(embedding=embedding, task=task, predict_subset=predict_subset)
        if persist_path is not None:
            evaluator = TextClassificationEvaluator().persisting(JsonPersister(path=persist_path))
        else:
            evaluator = TextClassificationEvaluator()

        super().__init__(dataset, data_loader, model, evaluator)


class FlairSequenceLabelingEvaluationPipeline(
    ModelEvaluationPipeline[str, Corpus, Predictions, SequenceLabelingEvaluationResults]
):
    DEFAULT_EVAL_MODE = EvaluationMode.CONLL

    def __init__(
        self,
        dataset_path: str,
        embedding_name: T_path,
        output_path: T_path,
        model_type_reference: str = "",
        evaluation_mode: EvaluationMode = DEFAULT_EVAL_MODE,
        tagging_scheme: Optional[TaggingScheme] = None,
        config: FlairSequenceLabelingConfig = FlairSequenceLabelingBasicConfig(),
        persist_path: Optional[T_path] = None,
        predict_subset: Literal["dev", "test"] = "test",
    ):
        dataset = Dataset(dataset=dataset_path)
        data_loader = ConllFlairCorpusDataLoader()

        embedding_loader = FlairWordEmbeddingLoader(embedding_name, model_type_reference)
        embedding = embedding_loader.get_embedding(**config.load_model_kwargs)

        task = SequenceLabeling(
            output_path,
            hidden_size=config.hidden_size,
            task_model_kwargs=config.task_model_kwargs,
            task_train_kwargs=config.task_train_kwargs,
        )
        model = FlairModel(embedding=embedding, task=task, predict_subset=predict_subset)
        if persist_path is not None:
            evaluator = SequenceLabelingEvaluator(
                evaluation_mode=evaluation_mode, tagging_scheme=tagging_scheme
            ).persisting(JsonPersister(path=persist_path))
        else:
            evaluator = SequenceLabelingEvaluator(
                evaluation_mode=evaluation_mode, tagging_scheme=tagging_scheme
            )

        super().__init__(dataset, data_loader, model, evaluator)
