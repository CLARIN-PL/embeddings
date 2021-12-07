import abc
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from transformers import AutoConfig, AutoModel

from embeddings.data.datamodule import HuggingFaceDataset


class HuggingFaceLightningTask(pl.LightningModule, abc.ABC):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        downstream_model_type: Type["AutoModel"],
        metrics: Optional[MetricCollection] = None,
        learning_rate: float = 1e-4,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        unfreeze_transformer_from_layer: Optional[int] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=num_labels, **config_kwargs if config_kwargs else {}
        )
        self.model = downstream_model_type.from_pretrained(model_name_or_path, config=self.config)
        self.configure_metrics(metrics=metrics)
        self.freeze_transformer()
        if unfreeze_transformer_from_layer is not None:
            self.unfreeze_transformer(unfreeze_from=unfreeze_transformer_from_layer)

    def configure_metrics(self, metrics: Optional[MetricCollection]) -> None:
        if metrics is None:
            metrics = self.get_default_metrics()
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def freeze_transformer(self) -> None:
        for param in self.model.base_model.parameters():
            param.requires_grad = False

    def unfreeze_transformer(self, unfreeze_from: int = -1) -> None:
        for name, param in self.model.base_model.named_parameters():
            if name.startswith("encoder.layer"):
                no_layer = int(name.split(".")[2])
                if no_layer >= unfreeze_from:
                    param.requires_grad = True

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
            assert self.trainer is not None
            train_loader = self.trainer.datamodule.train_dataloader()
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

    def predict(self, dataloader: DataLoader[HuggingFaceDataset]) -> Dict[str, np.ndarray]:
        predictions = torch.argmax(
            torch.cat([self.forward(**batch).logits for batch in dataloader]), dim=1
        ).numpy()
        assert isinstance(predictions, np.ndarray)
        ground_truth = torch.cat([x["labels"] for x in dataloader]).numpy()
        assert isinstance(ground_truth, np.ndarray)
        return {"y_pred": predictions, "y_true": ground_truth}
