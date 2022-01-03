from typing import Any, Dict

import pytorch_lightning as pl
import torch.cuda
from numpy import typing as nptyping
from torch.utils.data import DataLoader

from embeddings.data.datamodule import HuggingFaceDataModule
from embeddings.data.dataset import LightingDataModuleSubset
from embeddings.model.model import Model
from embeddings.task.lightning_task.lightning_task import HuggingFaceLightningTask


class LightningModel(Model[HuggingFaceDataModule, Dict[str, nptyping.NDArray[Any]]]):
    def __init__(
        self,
        trainer: pl.Trainer,
        task: HuggingFaceLightningTask,
        predict_subset: LightingDataModuleSubset,
    ) -> None:
        super().__init__()
        self.trainer = trainer
        self.task = task
        self.predict_subset = predict_subset

    def execute(self, data: HuggingFaceDataModule) -> Dict[str, nptyping.NDArray[Any]]:
        try:
            self.trainer.fit(self.task, data)
        except Exception as e:
            del self.trainer
            torch.cuda.empty_cache()  # type: ignore
            raise e

        dataloader = data.get_subset(subset=self.predict_subset)
        assert isinstance(dataloader, DataLoader)
        return self.task.predict(dataloader=dataloader)
