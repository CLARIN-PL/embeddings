from typing import Any, Dict, Literal

import pytorch_lightning as pl
import torch.cuda
from numpy import typing as nptyping
from torch.utils.data import DataLoader

from embeddings.data.dataset import LightingDataModuleSubset, get_subset_from_lighting_datamodule
from embeddings.model.model import Model
from embeddings.task.lightning_task.lightning_task import HuggingFaceLightningTask


class LightningModel(Model[pl.LightningDataModule, Dict[str, nptyping.NDArray[Any]]]):
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

    def execute(self, data: pl.LightningDataModule) -> Dict[str, nptyping.NDArray[Any]]:
        try:
            self.trainer.fit(self.task, data)
        except Exception as e:
            del self.trainer
            torch.cuda.empty_cache()  # type: ignore
            raise e

        dataloader = get_subset_from_lighting_datamodule(data=data, subset=self.predict_subset)
        assert isinstance(dataloader, DataLoader)
        return self.task.predict(dataloader=dataloader)
