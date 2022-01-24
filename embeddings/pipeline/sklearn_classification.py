from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
from typing_extensions import Literal

from embeddings.data.data_loader import HuggingFaceDataLoader
from embeddings.data.dataset import HuggingFaceDataset
from embeddings.embedding.sklearn_embedding import SklearnEmbedding
from embeddings.transformation.hf_transformation.to_pandas_transformation import (
    ToPandasHuggingFaceCorpusTransformation,
)
from embeddings.task.sklearn_task.text_classification import TextClassification


class SklearnClassificationPipeline:
    def __init__(
        self,
        dataset_name: str,
        input_column_name: str,
        target_column_name: str,
        model_type: Literal["tree", "forest", "logistic"] = "tree",
        fit_classifier_kwargs: Optional[Dict[str, Any]] = None,
        fit_vectorizer_kwargs: Optional[Dict[str, Any]] = None,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.dataset = HuggingFaceDataset(
            dataset_name, **load_dataset_kwargs if load_dataset_kwargs else {}
        )
        self.data_loader = HuggingFaceDataLoader()
        self.transformation = ToPandasHuggingFaceCorpusTransformation()
        self.embedding = SklearnEmbedding()
        self.task = TextClassification()

        self.input_column_name = input_column_name
        self.target_column_name = target_column_name
        self.fit_classifier_kwargs = fit_classifier_kwargs if fit_classifier_kwargs else {}
        self.fit_vectorizer_kwargs = fit_vectorizer_kwargs if fit_vectorizer_kwargs else {}
        self.task = TextClassification(
            model=model_type, train_model_kwargs=self.fit_classifier_kwargs
        )

    def run(self):
        data = self.data_loader.load(self.dataset)
        data = self.transformation.transform(data)
        for subset in data:
            data[subset].rename(
                {self.input_column_name: "x", self.target_column_name: "y"},
                axis="columns",
                inplace=True,
            )
        self.embedding.fit(data["train"]["x"].values, self.fit_vectorizer_kwargs)

        data_transformed = {}
        for subset in data:
            data_transformed[subset] = {
                "x": self.embedding.embed(data[subset]["x"].values),
                "y": data[subset]["y"].copy(),
            }

        result = self.task.fit_predict(data_transformed)
        return result
