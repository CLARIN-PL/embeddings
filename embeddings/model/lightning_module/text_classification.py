from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection
from transformers import AutoModelForSequenceClassification

from embeddings.data.io import T_path
from embeddings.model.lightning_module.huggingface_module import HuggingFaceLightningModule
from embeddings.utils.loggers import get_logger

_logger = get_logger(__name__)


class TextClassificationModule(HuggingFaceLightningModule):
    downstream_model_type = AutoModelForSequenceClassification

    def __init__(
        self,
        model_name_or_path: T_path,
        num_classes: int,
        finetune_last_n_layers: int,
        metrics: Optional[MetricCollection] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model_name_or_path=model_name_or_path,
            downstream_model_type=self.downstream_model_type,
            num_classes=num_classes,
            finetune_last_n_layers=finetune_last_n_layers,
            metrics=metrics,
            config_kwargs=config_kwargs,
            task_model_kwargs=task_model_kwargs,
        )

    def shared_step(self, **batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.forward(**batch)
        loss, logits = outputs[:2]
        preds = torch.argmax(logits, dim=1)
        return loss, logits, preds

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        assert isinstance(self.hparams, dict)
        batch, batch_idx = args
        loss, logits, _ = self.shared_step(**batch)
        self.train_metrics(logits, batch["labels"])
        self.log("train/Loss", loss)
        if self.hparams["use_scheduler"]:
            assert self.trainer is not None
            last_lr = self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()
            self.log("train/BaseLR", last_lr[0], prog_bar=True)
            self.log("train/LambdaLR", last_lr[1], prog_bar=True)
        return {"loss": loss}

    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        batch, batch_idx = args
        loss, logits, _ = self.shared_step(**batch)
        self.val_metrics.update(logits, batch["labels"])
        self.log("val/Loss", loss, on_epoch=True)
        return None

    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        batch, batch_idx = args
        loss, logits, _ = self.shared_step(**batch)
        if -1 not in batch["labels"]:
            self.test_metrics.update(logits, batch["labels"])
            self.log("test/Loss", loss, on_epoch=True)
        else:
            _logger.warning("Missing labels for the test data")
        return None
