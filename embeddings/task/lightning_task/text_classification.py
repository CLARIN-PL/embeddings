from typing import Any, Dict, Optional, Tuple, Type

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import F1, Accuracy, MetricCollection, Precision, Recall
from transformers import AutoModel, AutoModelForSequenceClassification

from embeddings.task.lightning_task.lightning_task import HuggingFaceLightningTask


class TextClassificationTask(HuggingFaceLightningTask):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        downstream_model_type: Type["AutoModel"] = AutoModelForSequenceClassification,
        metrics: Optional[MetricCollection] = None,
        learning_rate: float = 1e-4,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        config_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name_or_path=model_name_or_path,
            num_labels=num_labels,
            downstream_model_type=downstream_model_type,
            metrics=metrics,
            learning_rate=learning_rate,
            adam_epsilon=adam_epsilon,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            config_kwargs=config_kwargs,
            **kwargs,
        )

    def get_default_metrics(self) -> MetricCollection:
        if self.hparams.num_labels > 2:
            metrics = MetricCollection(
                [
                    Accuracy(num_classes=self.hparams.num_labels),
                    Precision(num_classes=self.hparams.num_labels, average="macro"),
                    Recall(num_classes=self.hparams.num_labels, average="macro"),
                    F1(num_classes=self.hparams.num_labels, average="macro"),
                ]
            )
        else:
            metrics = MetricCollection(
                [
                    Accuracy(num_classes=self.hparams.num_labels),
                    Precision(num_classes=self.hparams.num_labels),
                    Recall(num_classes=self.hparams.num_labels),
                    F1(num_classes=self.hparams.num_labels),
                ]
            )
        return metrics

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def shared_step(self, **batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.forward(**batch)
        loss, logits = outputs[:2]
        return loss, logits

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        batch, batch_idx = args
        loss, preds = self.shared_step(**batch)
        self.train_metrics(preds, batch["labels"])
        self.log("train/Loss", loss, on_step=True, on_epoch=True)
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
        return None
