from typing import Any, Dict, Optional

import datasets
import pandas as pd
from sklearn.base import BaseEstimator as AnySklearnVectorizer
from sklearn.base import ClassifierMixin as AnySklearnClassifier
from sklearn.feature_extraction.text import CountVectorizer
from typing_extensions import Literal

from embeddings.data.data_loader import HuggingFaceDataLoader
from embeddings.data.dataset import HuggingFaceDataset
from embeddings.data.io import T_path
from embeddings.embedding.sklearn_embedding import SklearnEmbedding
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.sklearn_model import SklearnModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.sklearn_task.text_classification import TextClassification
from embeddings.transformation.hf_transformation.to_pandas_transformation import (
    ToPandasHuggingFaceCorpusTransformation,
)
from embeddings.utils.json_dict_persister import JsonPersister


class SklearnClassificationPipeline(
    StandardPipeline[
        str,
        datasets.DatasetDict,
        Dict[str, pd.DataFrame],
        Dict[str, pd.DataFrame],
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
        vectorizer: AnySklearnVectorizer = CountVectorizer,
        evaluation_filename: str = "evaluation.json",
        predict_subset: Literal["dev", "validation", "test"] = "test",
        classifier_kwargs: Optional[Dict[str, Any]] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        dataset = HuggingFaceDataset(
            dataset_name, **load_dataset_kwargs if load_dataset_kwargs else {}
        )
        data_loader = HuggingFaceDataLoader()
        transformation = ToPandasHuggingFaceCorpusTransformation()
        classifier_kwargs = classifier_kwargs if classifier_kwargs else {}
        embedding = SklearnEmbedding(embedding_kwargs, vectorizer=vectorizer)
        task = TextClassification(classifier=classifier, classifier_kwargs=classifier_kwargs)
        model = SklearnModel(embedding, task, predict_subset=predict_subset)
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

        model_result = {
            "y_pred": self.model.execute(data),
            "y_true": data[self.model.predict_subset]["y"].values,
        }
        evaluation_result = self.evaluator.evaluate(model_result)
        return evaluation_result
