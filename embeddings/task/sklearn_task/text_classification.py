from typing import Any, Dict, Optional

from sklearn.base import ClassifierMixin as AnySklearnClassifier

from embeddings.embedding.sklearn_embedding import SklearnEmbedding
from embeddings.evaluator.evaluation_results import Predictions
from embeddings.task.sklearn_task.sklearn_task import SklearnTask
from embeddings.utils.array_like import ArrayLike


class TextClassification(SklearnTask):
    def __init__(
        self,
        classifier: AnySklearnClassifier,
        classifier_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.classifier_kwargs = classifier_kwargs if classifier_kwargs else {}
        self.classifier = classifier(**self.classifier_kwargs)

    def build_task_model(self, embedding: SklearnEmbedding) -> None:
        self.embedding = embedding

    def fit(
        self,
        data: Dict[str, ArrayLike],
    ) -> None:
        self.classifier.fit(
            self.embedding.embed(data["train"]["x"]),
            data["train"]["y"],
        )

    def predict(
        self,
        data: Dict[str, ArrayLike],
        predict_subset: str = "test",
    ) -> Predictions:
        predictions = self.classifier.predict(self.embedding.embed(data[predict_subset]["x"]))
        model_result = Predictions(y_pred=predictions, y_true=data[predict_subset]["y"].values)
        return model_result

    def fit_predict(
        self,
        data: Dict[str, ArrayLike],
        predict_subset: str = "test",
    ) -> Predictions:
        self.fit(data)
        return self.predict(data)
