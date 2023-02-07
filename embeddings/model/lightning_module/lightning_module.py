import abc
import inspect
from inspect import signature
from typing import Any, Dict, Generic, List, Literal, Optional, Tuple, TypeVar

import numpy as np
import pytorch_lightning as pl
import torch
from numpy import typing as nptyping
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn.functional import softmax
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall
from transformers import get_linear_schedule_with_warmup

from embeddings.data.datamodule import HuggingFaceDataset
from embeddings.utils.loggers import get_logger

Model = TypeVar("Model")

_logger = get_logger(__name__)


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
        assert inspect.ismethod(self.save_hyperparameters)
        self.save_hyperparameters(ignore=["downstream_model_type"])  # cannot pickle model type
        self.metrics = metrics

    @abc.abstractmethod
    def _init_model(self) -> None:
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
        logits, predictions = zip(*self._predict_with_trainer(dataloader))
        probabilities = softmax(torch.cat(logits), dim=1).numpy()
        predictions = torch.cat(predictions).numpy()
        ground_truth = torch.cat([x["labels"] for x in dataloader]).numpy()
        result = {"y_pred": predictions, "y_true": ground_truth, "y_probabilities": probabilities}
        assert all(isinstance(x, np.ndarray) for x in result.values())
        return result

    def _predict_with_trainer(self, dataloader: DataLoader[HuggingFaceDataset]) -> torch.Tensor:
        assert self.trainer is not None
        try:
            return self.trainer.predict(
                model=self, dataloaders=dataloader, return_predictions=True, ckpt_path="best"
            )
        except MisconfigurationException:  # model loaded but not fitted
            _logger.warning(
                "The best model checkpoint cannot be loaded because trainer.fit has not been called. Using current weights for prediction."
            )
            return self.trainer.predict(
                model=self,
                dataloaders=dataloader,
                return_predictions=True,
            )

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

    def _init_metrics(self) -> None:
        if self.metrics is None:
            self.metrics = self.get_default_metrics()
        self.train_metrics = self.metrics.clone(prefix="train/")
        self.val_metrics = self.metrics.clone(prefix="val/")
        self.test_metrics = self.metrics.clone(prefix="test/")

    def get_default_metrics(self) -> MetricCollection:
        assert isinstance(self.hparams, dict)

        task: Literal["multiclass", "binary"] = (
            "multiclass" if self.hparams["num_classes"] > 2 else "binary"
        )

        # Accuracy, Precision etc. no longer inherits from Metric subclass as it is helper class
        return MetricCollection(
            [
                Accuracy(num_classes=self.hparams["num_classes"], task=task),  # type: ignore[list-item]
                Precision(
                    num_classes=self.hparams["num_classes"], average="macro", task=task
                ),  # type: ignore[list-item]
                Recall(
                    num_classes=self.hparams["num_classes"], average="macro", task=task
                ),  # type: ignore[list-item]
                F1Score(
                    num_classes=self.hparams["num_classes"], average="macro", task=task
                ),  # type: ignore[list-item]
            ]
        )

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Any]]:
        """Prepare optimizer and schedule (linear warmup and decay)"""
        assert isinstance(self.hparams, dict)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams["weight_decay"],
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = getattr(torch.optim, self.hparams["optimizer"])
        assert "lr" in signature(optimizer_cls).parameters
        assert "eps" in signature(optimizer_cls).parameters
        optimizer = optimizer_cls(
            optimizer_grouped_parameters,
            lr=self.hparams["learning_rate"],
            eps=self.hparams["adam_epsilon"],
        )

        if self.hparams["use_scheduler"]:
            lr_schedulers = self._get_schedulers(optimizer=optimizer)
        else:
            lr_schedulers = []

        return [optimizer], lr_schedulers

    def _get_schedulers(self, optimizer: Optimizer) -> List[Dict[str, Any]]:
        assert isinstance(self.hparams, dict)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams["warmup_steps"],
            num_training_steps=self.total_steps,
        )
        return [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
