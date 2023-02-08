import sys
from collections import ChainMap
from typing import Dict, Optional, Union, Sequence, List, Any, Type

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering, AutoConfig, AutoModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from embeddings.data.dataset import Dataset
from embeddings.data.io import T_path
from embeddings.model.lightning_module.huggingface_module import HuggingFaceLightningModule
from embeddings.model.lightning_module.lightning_module import LightningModule

HuggingFaceDataset = Type[Dataset]  # to refactor


class PretrainedQAModel(pl.LightningModule):  # type: ignore
    """
    TODO:
    Refactor pt. 2
    Refactor in separate PR (create seperate task for inference)
    https://github.com/CLARIN-PL/embeddings/issues/279
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.trainer = pl.Trainer(devices="auto", accelerator="auto")

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
    ) -> List[Dict[str, Union[QuestionAnsweringModelOutput, Dict[str, torch.Tensor]]]]:
        assert self.trainer is not None
        return self.trainer.predict(  # type: ignore
            model=self,
            dataloaders=dataloaders,
            return_predictions=True,
        )

class QuestionAnsweringModule(LightningModule[AutoModelForQuestionAnswering]):
    """
    TODO:
    Refactor pt. 2:
    - Refactor functions `setup`, `forward` `freeze_transformer`
    """

    downstream_model_type = AutoModelForQuestionAnswering

    def __init__(
            self,
            model_name_or_path: T_path,
            finetune_last_n_layers: int,
            config_kwargs: Optional[Dict[str, Any]] = None,
            task_model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__(metrics=None, **task_model_kwargs if task_model_kwargs else {})
        self.save_hyperparameters({"downstream_model_type": self.downstream_model_type.__name__})
        self.downstream_model_type = self.downstream_model_type
        self.config_kwargs = config_kwargs if config_kwargs else {}
        self.target_names: Optional[List[str]] = None
        self._init_model()

    def _init_metrics(self) -> None:
        pass

    def _init_model(self) -> None:
        self.config = AutoConfig.from_pretrained(
            self.hparams.model_name_or_path,
            **self.config_kwargs,
        )
        self.model: AutoModel = self.downstream_model_type.from_pretrained(
            self.hparams.model_name_or_path, config=self.config
        )
        if self.hparams.finetune_last_n_layers > -1:
            self.freeze_transformer(finetune_last_n_layers=self.hparams.finetune_last_n_layers)

    def freeze_transformer(self, finetune_last_n_layers: int) -> None:
        """Borrowed from clarinpl-embeddings library"""
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
            if self.hparams.use_scheduler:
                train_loader = self.trainer.datamodule.train_dataloader()
                gpus = getattr(self.trainer, "gpus") if getattr(self.trainer, "gpus") else 0
                tb_size = self.hparams.train_batch_size * max(1, gpus)
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
        return {"data": batch, "outputs": outputs}

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        batch, batch_idx = args
        outputs = self(**batch)

        self.log("train/Loss", outputs.loss)
        if self.hparams.use_scheduler:
            assert self.trainer is not None
            last_lr = self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()
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

    def training_epoch_end(self, outputs: List[Any]) -> None:
        pass

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        pass

    def test_epoch_end(self, outputs: List[Any]) -> None:
        pass
