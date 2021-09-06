from typing import Any, Dict, Optional

from flair.data import Dictionary
from flair.models import TextPairClassifier

from embeddings.data.io import T_path
from embeddings.embedding.flair_embedding import FlairEmbedding
from embeddings.task.flair_task.text_classification import TextClassification


class TextPairClassification(TextClassification):
    def __init__(
        self,
        output_path: T_path,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(output_path, task_model_kwargs, task_train_kwargs)

    def build_task_model(self, embedding: FlairEmbedding, y_dictionary: Dictionary) -> None:
        self.model = TextPairClassifier(
            document_embeddings=embedding.model,
            label_dictionary=y_dictionary,
            label_type="class",
            **self.task_model_kwargs
        )
