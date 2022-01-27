from typing import Dict, Any, Union

from numpy import typing as nptyping
import pandas as pd
from typing_extensions import Literal

from embeddings.model.model import Model
from embeddings.task.sklearn_task.text_classification import TextClassification
from embeddings.embedding.sklearn_embedding import SklearnEmbedding


class SklearnModel(Model[Union[pd.DataFrame, nptyping.NDArray[Any]], Dict]):
    def __init__(
        self,
        embedding: SklearnEmbedding,
        task: TextClassification,
        predict_subset: Literal["dev", "validation", "test"] = "test",
    ):
        super().__init__()
        self.embedding = embedding
        self.task = task
        self.predict_subset = predict_subset

    def execute(self, data: Dict[str, any]):
        self.embedding.fit(data["train"]["x"])
        self.task.build_task_model(self.embedding)
        return self.task.fit_predict(data, self.predict_subset)
