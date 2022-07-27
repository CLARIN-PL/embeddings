from pathlib import Path
from typing import Any, Dict, Optional, Union

import datasets
import pandas as pd
from sklearn.base import BaseEstimator as AnySklearnVectorizer
from sklearn.base import ClassifierMixin as AnySklearnClassifier
from sklearn.feature_extraction.text import CountVectorizer
from typing_extensions import Literal

from embeddings.data.data_loader import (
    HuggingFaceDataLoader,
    HuggingFaceLocalDataLoader,
    get_hf_dataloader,
)
from embeddings.data.dataset import Dataset
from embeddings.data.io import T_path
from embeddings.embedding.sklearn_embedding import SklearnEmbedding
from embeddings.evaluator.evaluation_results import Predictions, TextClassificationEvaluationResults
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.sklearn_model import SklearnModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.sklearn_task.text_classification import TextClassification
from embeddings.transformation.hf_transformation.to_pandas_transformation import (
    ToPandasHuggingFaceCorpusTransformation,
)
from embeddings.transformation.pandas_transformation.rename_input_columns_transformation import (
    RenameInputColumnsTransformation,
)
from embeddings.utils.json_dict_persister import JsonPersister


class SklearnClassificationPipeline(
    StandardPipeline[
        str,
        datasets.DatasetDict,
        Dict[str, pd.DataFrame],
        Predictions,
        TextClassificationEvaluationResults,
    ]
):
    def __init__(
        self,
        dataset_name_or_path: T_path,
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
        dataset = Dataset(
            str(dataset_name_or_path), **load_dataset_kwargs if load_dataset_kwargs else {}
        )
        data_loader: Union[HuggingFaceDataLoader, HuggingFaceLocalDataLoader] = get_hf_dataloader(
            dataset
        )
        transformation = ToPandasHuggingFaceCorpusTransformation().then(
            RenameInputColumnsTransformation(input_column_name, target_column_name)
        )
        classifier_kwargs = classifier_kwargs if classifier_kwargs else {}
        embedding = SklearnEmbedding(embedding_kwargs=embedding_kwargs, vectorizer=vectorizer)
        task = TextClassification(classifier=classifier, classifier_kwargs=classifier_kwargs)
        model = SklearnModel(embedding, task, predict_subset=predict_subset)
        output_path = Path(output_path)
        evaluator = TextClassificationEvaluator().persisting(
            JsonPersister(path=output_path.joinpath(evaluation_filename))
        )
        super().__init__(dataset, data_loader, transformation, model, evaluator)

        self.input_column_name = input_column_name
        self.target_column_name = target_column_name
        self.predict_subset = predict_subset
