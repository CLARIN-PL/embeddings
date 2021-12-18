from collections import ChainMap
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from numpy import typing as nptyping
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader
from torchmetrics import F1, Accuracy, MetricCollection, Precision, Recall
from transformers import AutoModelForSequenceClassification

from embeddings.data.datamodule import HuggingFaceDataset
from embeddings.task.lightning_task.lightning_task import HuggingFaceLightningTask
from embeddings.utils.loggers import get_logger

_logger = get_logger(__name__)


class TextClassification(HuggingFaceLightningTask):
    downstream_model_type = AutoModelForSequenceClassification

    def __init__(
        self,
        model_name_or_path: str,
        finetune_last_n_layers: int = -1,
        metrics: Optional[MetricCollection] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            model_name_or_path=model_name_or_path,
            downstream_model_type=self.downstream_model_type,
            finetune_last_n_layers=finetune_last_n_layers,
            metrics=metrics,
            config_kwargs=config_kwargs,
            task_model_kwargs=task_model_kwargs,
            **kwargs
        )

    def get_default_metrics(self) -> MetricCollection:
        assert self.trainer is not None
        num_classes = self.trainer.datamodule.num_classes
        if num_classes > 2:
            metrics = MetricCollection(
                [
                    Accuracy(num_classes=num_classes),
                    Precision(num_classes=num_classes, average="macro"),
                    Recall(num_classes=num_classes, average="macro"),
                    F1(num_classes=num_classes, average="macro"),
                ]
            )
        else:
            metrics = MetricCollection(
                [
                    Accuracy(num_classes=num_classes),
                    Precision(num_classes=num_classes),
                    Recall(num_classes=num_classes),
                    F1(num_classes=num_classes),
                ]
            )
        return metrics

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        assert (not (args and kwargs)) and (args or kwargs)
        inputs = kwargs if kwargs else args
        if isinstance(inputs, tuple):
            inputs = dict(ChainMap(*inputs))
        return self.model(**inputs)

    def shared_step(self, **batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.forward(**batch)
        loss, logits = outputs[:2]
        return loss, logits

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        batch, batch_idx = args
        loss, preds = self.shared_step(**batch)
        self.train_metrics(preds, batch["labels"])
        self.log("train/Loss", loss)
        if self.hparams.use_scheduler:
            assert self.trainer is not None
            last_lr = self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()
            self.log("train/BaseLR", last_lr[0], prog_bar=True)
            self.log("train/LambdaLR", last_lr[1], prog_bar=True)
        return {"loss": loss}

    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        batch, batch_idx = args
        loss, preds = self.shared_step(**batch)
        self.val_metrics(preds, batch["labels"])
        self.log("val/Loss", loss, on_epoch=True)
        return None

    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        batch, batch_idx = args
        loss, preds = self.shared_step(**batch)
        if -1 not in batch["labels"]:
            self.test_metrics(preds, batch["labels"])
            self.log("test/Loss", loss, on_epoch=True)
        else:
            _logger.warning("Missing labels for the test data")
        return None

    def predict(
        self, dataloader: DataLoader[HuggingFaceDataset]
    ) -> Dict[str, nptyping.NDArray[Any]]:
        predictions = torch.argmax(
            torch.cat([self.forward(**batch).logits for batch in dataloader]), dim=1
        ).numpy()
        assert isinstance(predictions, np.ndarray)
        ground_truth = torch.cat([x["labels"] for x in dataloader]).numpy()
        assert isinstance(ground_truth, np.ndarray)
        return {"y_pred": predictions, "y_true": ground_truth}
