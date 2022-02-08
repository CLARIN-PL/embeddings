import abc
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from numpy import typing as nptyping
from torch.utils.data import DataLoader

from embeddings.callbacks.lightning_callbacks import BestEpochCallback
from embeddings.data.datamodule import HuggingFaceDataModule
from embeddings.data.dataset import LightingDataModuleSubset
from embeddings.data.io import T_path
from embeddings.model.lightning_module.huggingface_module import HuggingFaceLightningModule
from embeddings.task.task import Task
from embeddings.utils.loggers import get_logger

_logger = get_logger(__name__)


class LightningTask(Task[HuggingFaceDataModule, Dict[str, nptyping.NDArray[Any]]]):
    MODEL_UNDEFINED_EXCEPTION = ValueError("Model undefined. Use build_task_model() first!")

    def __init__(
        self,
        output_path: T_path,
        task_train_kwargs: Dict[str, Any],
        early_stopping_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.output_path: Path = Path(output_path)
        self.task_train_kwargs = task_train_kwargs
        self.early_stopping_kwargs = early_stopping_kwargs
        self.model: Optional[HuggingFaceLightningModule] = None
        self.trainer: Optional[pl.Trainer] = None

    @property
    def best_epoch(self) -> Optional[float]:
        if self.trainer is None:
            return

        for callback in self.trainer.callbacks:
            if isinstance(callback, BestEpochCallback):
                return callback.best_epoch

    @property
    def best_validation_score(self) -> Optional[float]:
        if self.trainer is None:
            return

        for callback in self.trainer.callbacks:
            if isinstance(callback, BestEpochCallback):
                return callback.best_score

    def fit(
        self,
        data: HuggingFaceDataModule,
    ) -> None:
        if not self.model:
            raise self.MODEL_UNDEFINED_EXCEPTION
        self.trainer = pl.Trainer(
            default_root_dir=str(self.output_path),
            callbacks=[BestEpochCallback(), EarlyStopping(**self.early_stopping_kwargs)],
            **self.task_train_kwargs)
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
    ) -> Dict[str, nptyping.NDArray[Any]]:
        if not self.model:
            raise self.MODEL_UNDEFINED_EXCEPTION
        self.fit(data)
        dataloader = data.get_subset(subset=predict_subset)
        assert isinstance(dataloader, DataLoader)
        return self.predict(dataloader=dataloader)

    @abc.abstractmethod
    def build_task_model(self) -> None:
        pass
