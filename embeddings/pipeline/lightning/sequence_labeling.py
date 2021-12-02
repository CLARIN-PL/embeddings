from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from embeddings.data.huggingface_datamodule import HuggingFaceDataset, SequenceLabelingDataModule
from embeddings.data.io import T_path
from embeddings.evaluator.sequence_labeling_evaluator import SequenceLabelingEvaluator
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.lightning.auto_lightning import AutoTransformerForTokenClassification
from embeddings.pipeline.lightning.pipeline import LightningPipeline


class LightningSequenceLabelingPipeline(LightningPipeline):
    def __init__(
        self,
        embedding_name: str,
        dataset_name: str,
        input_column_name: str,
        target_column_name: str,
        output_path: T_path,
        evaluation_mode: str = "conll",
        tagging_scheme: Optional[str] = None,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.dm = SequenceLabelingDataModule(
            model_name_or_path=embedding_name,
            dataset_name=dataset_name,
            text_field=input_column_name,
            target_field=target_column_name,
            load_dataset_kwargs=load_dataset_kwargs,
        )
        self.dm.initalize()

        self.model = AutoTransformerForTokenClassification(
            model_name_or_path=embedding_name,
            input_dim=self.dm.input_dim,
            num_labels=self.dm.num_labels,
            **self.instantiate_kwargs(task_model_kwargs),
        )
        self.trainer = pl.Trainer(
            default_root_dir=output_path, **self.instantiate_kwargs(task_train_kwargs)
        )
        self.evaluator = SequenceLabelingEvaluator(
            evaluation_mode=evaluation_mode, tagging_scheme=tagging_scheme
        )

    def predict(self, dataloader: DataLoader[HuggingFaceDataset]) -> Dict[str, NDArray[np.string_]]:
        predictions = []

        for batch in dataloader:
            outputs = self.model(**batch)
            predictions.append(outputs.logits)

        predictions = list(torch.argmax(torch.cat(predictions), dim=2).numpy())
        ground_truth = list(torch.cat([x["labels"] for x in dataloader]).numpy())

        for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            predictions[i] = [
                self.dm.id_to_label[x] for x in list(pred[gt != self.model.IGNORE_INDEX])
            ]
            ground_truth[i] = [
                self.dm.id_to_label[x] for x in list(gt[gt != self.model.IGNORE_INDEX])
            ]

        return {"y_pred": np.array(predictions), "y_true": np.array(ground_truth)}

    def run(self) -> Dict[str, Any]:
        self.dm.setup("fit")
        self.trainer.fit(self.model, self.dm)
        self.trainer.test(datamodule=self.dm)
        model_result = self.predict(dataloader=self.dm.test_dataloader())
        metrics = self.evaluator.evaluate(model_result)
        return metrics
