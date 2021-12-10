import abc
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup

from embeddings.data.datamodule import HuggingFaceDataset

Model = TypeVar("Model")


class LightningTask(pl.LightningModule, abc.ABC, Generic[Model]):
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
    def shared_step(self, **batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
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

    @abc.abstractmethod
    def predict(self, dataloader: DataLoader[HuggingFaceDataset]) -> Dict[str, np.ndarray]:
        pass

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
            lr_scheduler = self.configure_scheduler(optimizer=optimizer)
        else:
            lr_scheduler = []

        return [optimizer], lr_scheduler

    def configure_scheduler(self, optimizer: Optimizer) -> List[Dict[str, Any]]:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        self.lr_scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [self.lr_scheduler]


class HuggingFaceLightningTask(LightningTask[AutoModel], abc.ABC):
    def __init__(
        self,
        model_name_or_path: str,
        downstream_model_type: Type["AutoModel"],
        unfreeze_transformer_from_layer: Optional[int] = None,
        metrics: Optional[MetricCollection] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            metrics=metrics,
            **task_model_kwargs if task_model_kwargs else {},
        )
        self.save_hyperparameters({"downstream_model_type": downstream_model_type.__name__})
        self.downstream_model_type = downstream_model_type
        self.config_kwargs = config_kwargs if config_kwargs else {}

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self.configure_model()
            self.configure_metrics()
            if self.hparams.use_scheduler:
                assert self.trainer is not None
                train_loader = self.trainer.datamodule.train_dataloader()
                tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
                ab_size = tb_size * self.trainer.accumulate_grad_batches
                self.total_steps: int = int(
                    (len(train_loader.dataset) / ab_size) * float(self.trainer.max_epochs)
                )

    def configure_model(self) -> None:
        assert self.trainer is not None
        self.config = AutoConfig.from_pretrained(
            self.hparams.model_name_or_path,
            num_labels=self.trainer.datamodule.num_classes,
            **self.config_kwargs,
        )
        self.model: AutoModel = self.downstream_model_type.from_pretrained(
            self.hparams.model_name_or_path, config=self.config
        )
        self.freeze_transformer()
        if self.hparams.unfreeze_transformer_from_layer is not None:
            self.unfreeze_transformer(unfreeze_from=self.hparams.unfreeze_transformer_from_layer)

    def freeze_transformer(self) -> None:
        for param in self.model.base_model.parameters():
            param.requires_grad = False

    def unfreeze_transformer(self, unfreeze_from: int = -1) -> None:
        for name, param in self.model.base_model.named_parameters():
            if name.startswith("encoder.layer"):
                no_layer = int(name.split(".")[2])
                if no_layer >= unfreeze_from:
                    param.requires_grad = True