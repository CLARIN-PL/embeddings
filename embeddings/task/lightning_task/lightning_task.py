import abc
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
import wandb
from numpy import typing as nptyping
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import LightningLoggerBase
from torch.utils.data import DataLoader

from embeddings.data.datamodule import HuggingFaceDataModule
from embeddings.data.dataset import LightingDataModuleSubset
from embeddings.data.io import T_path
from embeddings.model.lightning_module.huggingface_module import HuggingFaceLightningModule
from embeddings.task.task import Task
from embeddings.utils.lightning_callbacks.best_epoch_callback import BestEpochCallback
from embeddings.utils.loggers import get_logger

_logger = get_logger(__name__)


class LightningTask(Task[HuggingFaceDataModule, Dict[str, nptyping.NDArray[Any]]]):
    MODEL_UNDEFINED_EXCEPTION = ValueError("Model undefined. Use build_task_model() first!")

    def __init__(
        self,
        output_path: T_path,
        task_train_kwargs: Dict[str, Any],
        early_stopping_kwargs: Dict[str, Any],
        logging_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.output_path: Path = Path(output_path)
        self.task_train_kwargs = task_train_kwargs
        self.early_stopping_kwargs = early_stopping_kwargs
        self.logging_kwargs = logging_kwargs
        self.model: Optional[HuggingFaceLightningModule] = None
        self.trainer: Optional[pl.Trainer] = None

    @property
    def best_epoch(self) -> Optional[float]:
        if self.trainer is None:
            return None

        for callback in self.trainer.callbacks:
            if isinstance(callback, BestEpochCallback):
                return callback.best_epoch
        return None

    @property
    def best_validation_score(self) -> Optional[float]:
        if self.trainer is None:
            return None

        for callback in self.trainer.callbacks:
            if isinstance(callback, BestEpochCallback):
                return callback.best_score.item()
        return None

    def fit(
        self,
        data: HuggingFaceDataModule,
        run_name: Optional[str] = None,
    ) -> None:
        if not self.model:
            raise self.MODEL_UNDEFINED_EXCEPTION

        callbacks: List[Callback] = []
        if "validation" in data.load_dataset().keys():
            callbacks.append(BestEpochCallback())
            callbacks.append(EarlyStopping(**self.early_stopping_kwargs))

        self.trainer = pl.Trainer(
            default_root_dir=str(self.output_path),
            callbacks=callbacks,
            logger=self.get_lightning_loggers(run_name, **self.logging_kwargs),
            **self.task_train_kwargs
        )
        try:
            self.trainer.fit(self.model, data)
        except Exception as e:
            del self.trainer
            torch.cuda.empty_cache()  # type: ignore
            raise e

    @abc.abstractmethod
    def predict(self, dataloader: DataLoader[Any]) -> Dict[str, nptyping.NDArray[Any]]:
        pass

    def fit_predict(
        self,
        data: HuggingFaceDataModule,
        predict_subset: LightingDataModuleSubset = LightingDataModuleSubset.TEST,
        run_name: Optional[str] = None,
    ) -> Dict[str, nptyping.NDArray[Any]]:
        if not self.model:
            raise self.MODEL_UNDEFINED_EXCEPTION
        self.fit(data, run_name=run_name)
        dataloader = data.get_subset(subset=predict_subset)
        assert isinstance(dataloader, DataLoader)
        result = self.predict(dataloader=dataloader)
        wandb.finish()
        return result

    @abc.abstractmethod
    def build_task_model(self) -> None:
        pass

    def get_lightning_loggers(
        self,
        run_name: Optional[str] = None,
        use_wandb: bool = True,
        use_tensorboard: bool = True,
        use_csv: bool = True,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
    ) -> List[LightningLoggerBase]:
        """Based on configuration, provides pytorch-lightning loggers' callbacks."""
        loggers: List[LightningLoggerBase] = []

        if use_tensorboard:
            loggers.append(
                pl_loggers.TensorBoardLogger(
                    name=run_name,
                    save_dir=str(self.output_path.joinpath("tensorboard")),
                )
            )

        if not use_wandb and (wandb_project or wandb_entity):
            raise ValueError(
                "`wandb_project` or `wandb_entity` was configured but use_wand is set to false."
            )

        if use_wandb:
            save_dir = self.output_path.joinpath("wandb")
            save_dir.mkdir(exist_ok=True)
            loggers.append(
                pl_loggers.WandbLogger(
                    name=run_name,
                    save_dir=str(save_dir),
                    project=wandb_project,
                    entity=wandb_entity,
                    reinit=True,
                )
            )

        if use_csv:
            loggers.append(
                pl_loggers.CSVLogger(name=run_name, save_dir=str(self.output_path.joinpath("csv")))
            )

        return loggers
