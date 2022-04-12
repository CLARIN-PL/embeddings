from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection
from transformers import AutoModelForTokenClassification

from embeddings.data.io import T_path
from embeddings.model.lightning_module.huggingface_module import HuggingFaceLightningModule
from embeddings.utils.loggers import get_logger

_logger = get_logger(__name__)


class SequenceLabelingModule(HuggingFaceLightningModule):
    downstream_model_type = AutoModelForTokenClassification

    def __init__(
        self,
        model_name_or_path: T_path,
        num_classes: int,
        finetune_last_n_layers: int,
        metrics: Optional[MetricCollection] = None,
        ignore_index: int = -100,
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
        self.ignore_index = ignore_index
        self._str2int: Optional[Dict[str, int]] = None
        self._int2str: Optional[Dict[int, str]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            assert self.trainer is not None
            self._int2str = (
                self.trainer.datamodule.dataset["train"].features["labels"].feature._int2str
            )
            self._str2int = (
                self.trainer.datamodule.dataset["train"].features["labels"].feature._str2int
            )
        super().setup(stage=stage)

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
                preds[labels != self.ignore_index], labels[labels != self.ignore_index]
            )
            self.log("test/Loss", loss, on_epoch=True)
        else:
            _logger.warning("Missing labels for the test data")
        return None

    def str2int(self, values: Union[str, Iterable[Any]]) -> Union[int, Iterable[Any]]:
        """Conversion class name string => integer duplicated from huggingface ClassLabel."""
        assert isinstance(values, str) or isinstance(
            values, Iterable
        ), f"Values {values} should be a string or an Iterable (list, numpy array, pytorch, tensorflow tensors)"
        return_list = True
        if isinstance(values, str):
            values = [values]
            return_list = False

        output = []
        for value in values:
            if self._str2int:
                # strip key if not in dict
                if value not in self._str2int:
                    value = str(value).strip()
                output.append(self._str2int[str(value)])
            else:
                # No names provided, try to integerize
                failed_parse = False
                try:
                    output.append(int(value))
                    if not 0 <= int(value) < self.hparams.num_classes:
                        failed_parse = True
                except ValueError:
                    failed_parse = True
                if failed_parse:
                    raise ValueError(f"Invalid string class label {value}")
        return output if return_list else output[0]

    def int2str(self, values: Union[int, Iterable[Any]]) -> Union[str, Iterable[Any]]:
        """Conversion integer => class name string duplicated from huggingface ClassLabel."""
        assert isinstance(values, int) or isinstance(
            values, Iterable
        ), f"Values {values} should be an integer or an Iterable (list, numpy array, pytorch, tensorflow tensors)"
        return_list = True
        if isinstance(values, int):
            values = [values]
            return_list = False

        for v in values:
            if not 0 <= v < self.hparams.num_classes:
                raise ValueError(f"Invalid integer class label {v:d}")

        if self._int2str:
            output = [self._int2str[int(v)] for v in values]
        else:
            # No names provided, return str(values)
            output = [str(v) for v in values]
        return output if return_list else output[0]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["_int2str"] = self._int2str
        checkpoint["_str2int"] = self._str2int
        super().on_save_checkpoint(checkpoint=checkpoint)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self._int2str = checkpoint["_int2str"]
        self._str2int = checkpoint["_str2int"]
        super().on_load_checkpoint(checkpoint=checkpoint)
