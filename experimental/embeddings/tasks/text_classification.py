from typing import Tuple, Union, List, Optional, Dict, Any

import torch
from datasets import load_metric, Metric
from flair.data import Dictionary, FlairDataset, DataPoint
from flair.datasets import SentenceDataset, DataLoader
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from torch.utils.data.dataset import Dataset

from experimental.data.io import T_path
from experimental.embeddings.tasks.base_task import BaseTask


class FlairTextClassification(BaseTask):
    def __init__(
        self,
        embeddings: TransformerDocumentEmbeddings,
        label_dict: Dictionary,
        output_path: T_path,
        metrics: Optional[List[Tuple[Metric, Dict[str, Any]]]] = None,
    ):
        model = TextClassifier(
            document_embeddings=embeddings,
            label_dictionary=label_dict,
        )
        super().__init__(model, output_path, metrics)

    def evaluate(self, dataset: FlairDataset) -> List[Optional[Dict[Any, Any]]]:
        y_pred, y_true = self.predict(dataset)

        if any(len(predicted) > 1 for predicted in y_pred):
            raise NotImplementedError("Multilabel evaluation is not supported yet.")

        y_true = [self.model.label_dictionary.get_idx_for_item(labels[0]) for labels in y_true]
        y_pred = [self.model.label_dictionary.get_idx_for_item(labels[0]) for labels in y_pred]

        return self.compute_metrics(y_true, y_pred)

    def predict(
        self,
        sentences: Union[List[DataPoint], SentenceDataset],
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: int = 8,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """A part of a function from flair.models.text_classification_model.TextClassifier."""

        # read Dataset into data loader (if list of sentences passed, make Dataset first)
        if not isinstance(sentences, Dataset):
            sentences = SentenceDataset(sentences)
        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)

        # use scikit-learn to evaluate
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch in data_loader:
                # remove previously predicted labels
                [sentence.remove_labels("predicted") for sentence in batch]

                true_values = [sentence.get_labels(self.model.label_type) for sentence in batch]
                self.model.predict(
                    batch,
                    embedding_storage_mode=embedding_storage_mode,
                    mini_batch_size=mini_batch_size,
                    label_name="predicted",
                    return_loss=True,
                )
                predictions = [sentence.get_labels("predicted") for sentence in batch]

                y_pred = [[label.value for label in prediction] for prediction in predictions]
                y_true = [[label.value for label in true_value] for true_value in true_values]

            # remove predicted labels
            [sentence.remove_labels("predicted") for sentence in sentences]

        return y_pred, y_true

    def _get_default_metrics(self) -> List[Tuple[Metric, Dict[str, Any]]]:
        return [
            (load_metric("accuracy"), {}),
            (load_metric("f1"), {"average": "macro"}),
            (load_metric("recall"), {"average": "macro"}),
            (load_metric("precision"), {"average": "macro"}),
        ]
