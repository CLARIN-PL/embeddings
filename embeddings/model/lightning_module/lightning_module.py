import abc
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np
import pytorch_lightning as pl
import torch
from numpy import typing as nptyping
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from transformers import get_linear_schedule_with_warmup

from embeddings.data.datamodule import HuggingFaceDataset

Model = TypeVar("Model")


class LightningModule(pl.LightningModule, abc.ABC, Generic[Model]):
    def __init__(
        self,
        metrics: Optional[MetricCollection] = None,
        learning_rate: float = 1e-4,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 100,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        use_scheduler: bool = False,
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

    def predict_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        batch, batch_idx = args
        loss, logits, preds = self.shared_step(**batch)
        return preds

    def predict(
        self, dataloader: DataLoader[HuggingFaceDataset]
    ) -> Dict[str, nptyping.NDArray[Any]]:
        assert self.trainer is not None
        predictions = self.trainer.predict(dataloaders=dataloader, return_predictions=True)
        predictions = torch.cat(predictions).numpy()
        assert isinstance(predictions, np.ndarray)
        ground_truth = torch.cat([x["labels"] for x in dataloader]).numpy()
        assert isinstance(ground_truth, np.ndarray)
        return {"y_pred": predictions, "y_true": ground_truth}

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
        optimizer = AdamW(
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
