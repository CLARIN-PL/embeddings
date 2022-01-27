from typing import Dict, Optional, Any

from numpy import typing as nptyping
import pandas as pd
from sklearn.base import ClassifierMixin as AnySklearnClassifier
from typing_extensions import Literal
import datasets

from embeddings.data.data_loader import HuggingFaceDataLoader
from embeddings.data.dataset import HuggingFaceDataset
from embeddings.data.io import T_path
from embeddings.embedding.sklearn_embedding import SklearnEmbedding
from embeddings.transformation.hf_transformation.to_pandas_transformation import (
    ToPandasHuggingFaceCorpusTransformation,
)
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.sklearn_task.text_classification import TextClassification
from embeddings.model.sklearn_model import SklearnModel
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.utils.json_dict_persister import JsonPersister


class SklearnClassificationPipeline(
    StandardPipeline[
        str,
        datasets.DatasetDict,
        Dict[str, pd.DataFrame],
        Dict[str, nptyping.NDArray[Any]],
        Dict[str, Any],
    ]
):
    def __init__(
        self,
        dataset_name: str,
        input_column_name: str,
        target_column_name: str,
        output_path: T_path,
        classifier: AnySklearnClassifier,
        vectorizer: Optional[str] = "bow",
        evaluation_filename: str = "evaluation.json",
        fit_classifier_kwargs: Optional[Dict[str, Any]] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        dataset = HuggingFaceDataset(
            dataset_name, **load_dataset_kwargs if load_dataset_kwargs else {}
        )
        data_loader = HuggingFaceDataLoader()
        transformation = ToPandasHuggingFaceCorpusTransformation()
        fit_classifier_kwargs = fit_classifier_kwargs if fit_classifier_kwargs else {}
        embedding = SklearnEmbedding(embedding_kwargs, method=vectorizer)
        task = TextClassification(classifier=classifier, train_model_kwargs=fit_classifier_kwargs)
        model = SklearnModel(embedding, task)
        evaluator = TextClassificationEvaluator().persisting(
            JsonPersister(path=output_path.joinpath(evaluation_filename))
        )
        super().__init__(dataset, data_loader, transformation, model, evaluator)

        self.input_column_name = input_column_name
        self.target_column_name = target_column_name

    def run(self):
        data = self.data_loader.load(self.dataset)
        data = self.transformation.transform(data)
        for subset in data:
            data[subset].rename(
                {self.input_column_name: "x", self.target_column_name: "y"},
                axis="columns",
                inplace=True,
            )
        return self.model.execute(data)
