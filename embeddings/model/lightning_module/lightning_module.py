import abc
from inspect import signature
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np
import pytorch_lightning as pl
import torch
from numpy import typing as nptyping
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn.functional import softmax
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from transformers import get_linear_schedule_with_warmup

from embeddings.data.datamodule import HuggingFaceDataset

Model = TypeVar("Model")


class LightningModule(pl.LightningModule, abc.ABC, Generic[Model]):
    def __init__(
        self,
        optimizer: str,
        learning_rate: float,
        adam_epsilon: float,
        warmup_steps: int,
        weight_decay: float,
        train_batch_size: int,
        eval_batch_size: int,
        use_scheduler: bool,
        metrics: Optional[MetricCollection] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["downstream_model_type"])  # cannot pickle model type
        self.metrics = metrics

    @abc.abstractmethod
    def get_default_metrics(self) -> MetricCollection:
        pass

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abc.abstractmethod
    def shared_step(self, **batch: Any) -> Tuple[torch.Tensor, ...]:
        pass

    @abc.abstractmethod
    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        pass

    @abc.abstractmethod
    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        pass

    @abc.abstractmethod
    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        pass

    def predict_step(self, *args: Any, **kwargs: Any) -> Optional[Tuple[STEP_OUTPUT, STEP_OUTPUT]]:
        batch, batch_idx = args
        loss, logits, preds = self.shared_step(**batch)
        return logits, preds

    def predict(
        self, dataloader: DataLoader[HuggingFaceDataset]
    ) -> Dict[str, nptyping.NDArray[Any]]:
        assert self.trainer is not None
        logits_predictions = self.trainer.predict(
            dataloaders=dataloader, return_predictions=True, ckpt_path="best"
        )
        logits, predictions = zip(*logits_predictions)
        probabilities = softmax(torch.cat(logits), dim=1).numpy()
        predictions = torch.cat(predictions).numpy()
        ground_truth = torch.cat([x["labels"] for x in dataloader]).numpy()
        result = {"y_pred": predictions, "y_true": ground_truth, "y_probabilities": probabilities}
        assert all(isinstance(x, np.ndarray) for x in result.values())
        return result

    def configure_metrics(self) -> None:
        if self.metrics is None:
            self.metrics = self.get_default_metrics()
        self.train_metrics = self.metrics.clone(prefix="train/")
        self.val_metrics = self.metrics.clone(prefix="val/")
        self.test_metrics = self.metrics.clone(prefix="test/")

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

        optimizer_cls = getattr(torch.optim, self.hparams.optimizer)
        assert "lr" in signature(optimizer_cls).parameters
        assert "eps" in signature(optimizer_cls).parameters
        optimizer = optimizer_cls(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        if self.hparams.use_scheduler:
            lr_schedulers = self.configure_schedulers(optimizer=optimizer)
        else:
            lr_schedulers = []

        return [optimizer], lr_schedulers

    def configure_schedulers(self, optimizer: Optimizer) -> List[Dict[str, Any]]:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        return [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
