import abc
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import AdamW, Optimizer
from torchmetrics import F1, Accuracy, MetricCollection, Precision, Recall
from transformers import get_linear_schedule_with_warmup


class SequenceClassificationModule(pl.LightningModule, abc.ABC):
    def __init__(
        self,
        num_labels: int,
        metrics: Optional[MetricCollection] = None,
        learning_rate: float = 1e-4,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if metrics is None:
            metrics = self.get_default_metrics(num_labels=num_labels)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    @staticmethod
    def get_default_metrics(num_labels: int) -> MetricCollection:
        if num_labels > 2:
            metrics = MetricCollection(
                [
                    Accuracy(num_classes=num_labels),
                    Precision(num_classes=num_labels, average="macro"),
                    Recall(num_classes=num_labels, average="macro"),
                    F1(num_classes=num_labels, average="macro"),
                ]
            )
        else:
            metrics = MetricCollection(
                [
                    Accuracy(num_classes=num_labels),
                    Precision(num_classes=num_labels),
                    Recall(num_classes=num_labels),
                    F1(num_classes=num_labels),
                ]
            )
        return metrics

    def shared_step(self, **batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.forward(**batch)
        loss, logits = outputs[:2]
        preds = torch.argmax(logits, dim=1)
        # loss_fn = torch.nn.CrossEntropyLoss()
        # loss = loss_fn(logits.view(-1, self.hparams.num_labels), batch["labels"].view(-1))
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

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self._aggregate_and_log_metrics(self.train_metrics)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self._aggregate_and_log_metrics(self.val_metrics, prog_bar=True)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self._aggregate_and_log_metrics(self.test_metrics)

    def _aggregate_and_log_metrics(
        self, metrics: MetricCollection, prog_bar: bool = False
    ) -> Dict[str, float]:
        metric_values = metrics.compute()
        metrics.reset()
        self.log_dict(metric_values, prog_bar=prog_bar)
        return metric_values

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.trainer.datamodule.train_dataloader()

            # Calculate total steps
            tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
            ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
            self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Any]]:
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        return [optimizer], []
