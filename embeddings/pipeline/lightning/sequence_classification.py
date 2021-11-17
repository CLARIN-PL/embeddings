from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from embeddings.data.huggingface_datamodule import HuggingFaceDataset, TextClassificationDataModule
from embeddings.data.io import T_path
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.lightning.auto_lightning import AutoTransformerForSequenceClassification
from embeddings.pipeline.lightning.pipeline import LightningPipeline


class LightningClassificationPipeline(LightningPipeline):
    def __init__(
        self,
        embedding_name: str,
        dataset_name: str,
        input_column_name: Union[str, List[str]],
        target_column_name: str,
        output_path: T_path,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.dm = TextClassificationDataModule(
            model_name_or_path=embedding_name,
            dataset_name=dataset_name,
            text_fields=input_column_name,
            target_field=target_column_name,
            load_dataset_kwargs=load_dataset_kwargs,
        )
        self.dm.initalize()

        self.model = AutoTransformerForSequenceClassification(
            model_name_or_path=embedding_name,
            input_dim=self.dm.input_dim,
            num_labels=self.dm.num_labels,
            **self.instantiate_kwargs(task_model_kwargs),
        )
        self.trainer = pl.Trainer(
            default_root_dir=output_path, **self.instantiate_kwargs(task_train_kwargs)
        )
        self.evaluator = TextClassificationEvaluator()

    def predict(self, dataloader: DataLoader[HuggingFaceDataset]) -> Dict[str, np.ndarray]:
        predictions = []
        for batch in dataloader:
            outputs = self.model(**batch)
            predictions.append(outputs.logits)
        predictions = torch.argmax(torch.cat(predictions), dim=1).numpy()
        ground_truth = torch.cat([x["labels"] for x in dataloader]).numpy()
        return {"y_pred": predictions, "y_true": ground_truth}

    def run(self) -> Dict[str, Any]:
        self.dm.setup("fit")
        self.trainer.fit(self.model, self.dm)
        self.trainer.test(datamodule=self.dm)
        model_result = self.predict(dataloader=self.dm.test_dataloader())
        metrics = self.evaluator.evaluate(model_result)
        return metrics
