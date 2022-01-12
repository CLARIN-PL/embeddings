from collections import ChainMap
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from numpy import typing as nptyping
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader
from torchmetrics import F1, Accuracy, MetricCollection, Precision, Recall
from transformers import AutoModelForTokenClassification

from embeddings.data.datamodule import HuggingFaceDataset
from embeddings.task.lightning_task.lightning_task import HuggingFaceLightningTask
from embeddings.utils.loggers import get_logger
from embeddings.utils.utils import NDArrayInt

_logger = get_logger(__name__)


class SequenceLabeling(HuggingFaceLightningTask):
    downstream_model_type = AutoModelForTokenClassification

    def __init__(
        self,
        model_name_or_path: str,
        finetune_last_n_layers: int = -1,
        metrics: Optional[MetricCollection] = None,
        ignore_index: int = -100,
        config_kwargs: Optional[Dict[str, Any]] = None,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.ignore_index = ignore_index
        super().__init__(
            model_name_or_path=model_name_or_path,
            downstream_model_type=self.downstream_model_type,
            finetune_last_n_layers=finetune_last_n_layers,
            metrics=metrics,
            config_kwargs=config_kwargs,
            task_model_kwargs=task_model_kwargs,
            **kwargs,
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

    def shared_step(self, **batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.forward(**batch)
        loss, logits = outputs[:2]
        preds = torch.argmax(logits, dim=2)
        return loss, logits, preds

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        batch, batch_idx = args
        labels = batch["labels"]
        loss, logits, preds = self.shared_step(**batch)
        self.train_metrics(preds[labels != self.ignore_index], labels[labels != self.ignore_index])
        self.log("train/Loss", loss)
        if self.hparams.use_scheduler:
            assert self.trainer is not None
            last_lr = self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()
            self.log("train/BaseLR", last_lr[0], prog_bar=True)
            self.log("train/LambdaLR", last_lr[1], prog_bar=True)
        return {"loss": loss}

    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        batch, batch_idx = args
        labels = batch["labels"]
        loss, logits, preds = self.shared_step(**batch)
        self.val_metrics(preds[labels != self.ignore_index], labels[labels != self.ignore_index])
        self.log("val/Loss", loss, on_epoch=True)
        return None

    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        batch, batch_idx = args
        labels = batch["labels"]
        loss, logits, preds = self.shared_step(**batch)
        if -1 not in labels:
            self.test_metrics(
                preds[labels != self.IGNORE_INDEX], labels[labels != self.IGNORE_INDEX]
            )
            self.log("test/Loss", loss, on_epoch=True)
        else:
            _logger.warning("Missing labels for the test data")
        return None

    def predict_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        batch, batch_idx = args
        loss, logits, preds = self.shared_step(**batch)
        return preds

    def predict(
        self, dataloader: DataLoader[HuggingFaceDataset]
    ) -> Dict[str, nptyping.NDArray[np.str_]]:
        assert self.trainer is not None
        predictions = self.trainer.predict(
            dataloaders=dataloader, return_predictions=True
        )
        predictions = list(torch.cat(predictions).numpy())
        ground_truth = list(torch.cat([x["labels"] for x in dataloader]).numpy())

        def map_filter_data(data: NDArrayInt, ground_truth_data: NDArrayInt) -> List[str]:
            data = [
                self.trainer.datamodule.id2str(x.item())
                for x in data[ground_truth_data != self.ignore_index]
            ]
            return data

        for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            predictions[i] = map_filter_data(pred, gt)
            ground_truth[i] = map_filter_data(gt, gt)

        return {"y_pred": np.array(predictions), "y_true": np.array(ground_truth)}
