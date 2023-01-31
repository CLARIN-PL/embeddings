import abc
import inspect
import sys
from collections import ChainMap
from typing import Any, Dict, List, Optional, Type

from torchmetrics import MetricCollection
from transformers import AutoConfig, AutoModel

from embeddings.data.io import T_path
from embeddings.model.lightning_module.lightning_module import LightningModule


class HuggingFaceLightningModule(LightningModule[AutoModel], abc.ABC):
    def __init__(
        self,
        model_name_or_path: T_path,
        downstream_model_type: Type["AutoModel"],
        num_classes: int,
        finetune_last_n_layers: int,
        metrics: Optional[MetricCollection] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(metrics=metrics, **task_model_kwargs if task_model_kwargs else {})
        assert inspect.ismethod(self.save_hyperparameters)
        self.save_hyperparameters({"downstream_model_type": downstream_model_type.__name__})
        self.downstream_model_type = downstream_model_type
        self.config_kwargs = config_kwargs if config_kwargs else {}
        self.target_names: Optional[List[str]] = None
        self._init_model()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            assert self.trainer is not None
            self.target_names = self.trainer.datamodule.target_names
            if self.hparams.use_scheduler:
                train_loader = self.trainer.datamodule.train_dataloader()
                gpus = getattr(self.trainer, "gpus") if getattr(self.trainer, "gpus") else 0
                tb_size = self.hparams.train_batch_size * max(1, gpus)
                ab_size = tb_size * self.trainer.accumulate_grad_batches
                self.total_steps: int = int(
                    (len(train_loader.dataset) / ab_size) * float(self.trainer.max_epochs)
                )
            self._init_metrics()

    def _init_model(self) -> None:
        assert isinstance(self.hparams, dict)
        self.config = AutoConfig.from_pretrained(
            self.hparams["model_name_or_path"],
            num_labels=self.hparams["num_classes"],
            **self.config_kwargs,
        )
        self.model: AutoModel = self.downstream_model_type.from_pretrained(
            self.hparams["model_name_or_path"], config=self.config
        )
        if self.hparams["finetune_last_n_layers"] > -1:
            self.freeze_transformer(finetune_last_n_layers=self.hparams["finetune_last_n_layers"])

    def freeze_transformer(self, finetune_last_n_layers: int) -> None:
        if finetune_last_n_layers == 0:
            for name, param in self.model.base_model.named_parameters():
                param.requires_grad = False
        else:
            no_layers = self.model.config.num_hidden_layers
            for name, param in self.model.base_model.named_parameters():
                if name.startswith("embeddings"):
                    layer = 0
                elif name.startswith("encoder"):
                    layer = int(name.split(".")[2])
                elif name.startswith("pooler"):
                    layer = sys.maxsize
                else:
                    raise ValueError("Parameter name not recognized when freezing transformer")
                if layer >= (no_layers - finetune_last_n_layers):
                    break
                param.requires_grad = False

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        assert (not (args and kwargs)) and (args or kwargs)
        inputs = kwargs if kwargs else args
        if isinstance(inputs, tuple):
            inputs = dict(ChainMap(*inputs))
        return self.model(**inputs)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        assert self.trainer is not None
        checkpoint["target_names"] = self.trainer.datamodule.target_names

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.target_names = checkpoint["target_names"]
