from typing import Any, Dict

from typing_extensions import Literal

from embeddings.embedding.sklearn_embedding import SklearnEmbedding
from embeddings.evaluator.evaluation_results import Predictions
from embeddings.model.model import Model
from embeddings.task.sklearn_task.text_classification import TextClassification


class SklearnModel(Model[Dict[str, Any], Predictions]):
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

    def execute(self, data: Dict[str, Any], **kwargs: Any) -> Predictions:
        self.embedding.fit(data["train"]["x"])
        self.task.build_task_model(self.embedding)
        return self.task.fit_predict(data, self.predict_subset)
