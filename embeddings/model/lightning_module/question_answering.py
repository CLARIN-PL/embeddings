import sys
from collections import ChainMap
from typing import Any, Dict, List, Optional, Sequence, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import _PREDICT_OUTPUT, STEP_OUTPUT
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoModelForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from embeddings.data.datamodule import HuggingFaceDataset
from embeddings.data.io import T_path
from embeddings.model.lightning_module.lightning_module import LightningModule

QA_PREDICTED_ANSWER_TYPE = Dict[str, Union[str, int, float]]
QA_GOLD_ANSWER_TYPE = Dict[str, Union[Dict[str, Union[List[str], List[Any]]], str, int]]


class QuestionAnsweringInferenceModule(pl.LightningModule):
    def __init__(self, model_name: str, devices: str = "auto", accelerator: str = "auto") -> None:
        super().__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.trainer = pl.Trainer(devices=devices, accelerator=accelerator)

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Dict[str, Union[QuestionAnsweringModelOutput, Dict[str, torch.Tensor]]]:
        outputs = self.model(**batch)
        return {"data": batch, "outputs": outputs}

    def predict(
        self,
        dataloaders: Union[
            DataLoader[HuggingFaceDataset], Sequence[DataLoader[HuggingFaceDataset]]
        ],
    ) -> _PREDICT_OUTPUT:
        assert self.trainer is not None
        predict_output = self.trainer.predict(
            model=self,
            dataloaders=dataloaders,
            return_predictions=True,
        )
        assert predict_output
        return predict_output


class QuestionAnsweringModule(LightningModule[AutoModelForQuestionAnswering]):
    downstream_model_type = AutoModelForQuestionAnswering

    def __init__(
        self,
        model_name_or_path: T_path,
        finetune_last_n_layers: int,
        config_kwargs: Optional[Dict[str, Any]] = None,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        model_compile_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            metrics=None,
            **task_model_kwargs if task_model_kwargs else {},
        )
        # without type: ignore error: "Tensor" not callable  [operator]
        self.save_hyperparameters({"downstream_model_type": self.downstream_model_type.__name__})  # type: ignore
        self.downstream_model_type = self.downstream_model_type
        self.config_kwargs = config_kwargs if config_kwargs else {}
        self.target_names: Optional[List[str]] = None
        self.model_compile_kwargs = model_compile_kwargs
        self._init_model()

    def _init_metrics(self) -> None:
        pass

    def _init_model(self) -> None:
        self.config = AutoConfig.from_pretrained(
            # item "Tensor" of "Union[Tensor, Module]" has no attribute "model_name_or_path"
            self.hparams.model_name_or_path,  # type: ignore[union-attr]
            **self.config_kwargs,
        )
        self.model: AutoModel = self.downstream_model_type.from_pretrained(
            self.hparams.model_name_or_path, config=self.config  # type: ignore[union-attr]
        )
        if isinstance(self.model_compile_kwargs, dict):
            self.model = torch.compile(self.model, **self.model_compile_kwargs)
        # item "Tensor" of "Union[Tensor, Module]" has no attribute "finetune_last_n_layers"
        # unsupported operand type for <
        if self.hparams.finetune_last_n_layers > -1:  # type: ignore[union-attr, operator]
            # Argument "finetune_last_n_layers" to "freeze_transformer" of "QuestionAnsweringModule" has incompatible type "Union[Any, Tensor, Module]"; expected "int"
            self.freeze_transformer(finetune_last_n_layers=self.hparams.finetune_last_n_layers)  # type: ignore[union-attr, arg-type]

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

    def setup(self, stage: Optional[str] = None) -> None:
        """Borrowed from clarinpl-embeddings library"""
        if stage in ("fit", None):
            assert self.trainer is not None
            use_scheduler = getattr(self.hparams, "use_scheduler")
            if use_scheduler:
                datamodule = getattr(self.trainer, "datamodule")
                train_batch_size = getattr(self.hparams, "train_batch_size")
                if not self.trainer.max_epochs:
                    raise ValueError("Unable to retrieve max_epochs from trainer.")
                train_loader = datamodule.train_dataloader()
                tb_size = train_batch_size * max(1, self.trainer.num_devices)
                ab_size = tb_size * self.trainer.accumulate_grad_batches

                self.total_steps: int = int(
                    (len(train_loader.dataset) / ab_size) * float(self.trainer.max_epochs)
                )
            self._init_metrics()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Borrowed from clarinpl-embeddings library"""

        assert (not (args and kwargs)) and (args or kwargs)
        inputs = kwargs if kwargs else args
        if isinstance(inputs, tuple):
            inputs = dict(ChainMap(*inputs))
        return self.model(**inputs)

    def shared_step(self, **batch: Any) -> Any:
        outputs = self(**batch)
        return {"data": batch, "outputs": dict(outputs.items())}

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        batch, batch_idx = args
        outputs = self(**batch)

        self.log("train/Loss", outputs.loss)
        # without type: ignore mypy throws an error
        # Item "Tensor" of "Union[Tensor, Module]" has no attribute "use_scheduler"
        if self.hparams.use_scheduler:  # type: ignore[union-attr]
            assert self.trainer is not None
            last_lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()
            self.log("train/BaseLR", last_lr[0], prog_bar=True)
            self.log("train/LambdaLR", last_lr[1], prog_bar=True)
        return {"loss": outputs.loss}

    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        batch, batch_idx = args
        outputs = self(**batch)
        self.log("val/Loss", outputs.loss)
        return {"loss": outputs.loss}

    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        batch, batch_idx = args
        _ = self.shared_step(**batch)
        return None

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        batch, batch_idx = args
        return self.shared_step(**batch)

    def on_train_epoch_end(self) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        pass
