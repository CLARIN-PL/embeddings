from typing import Any, Dict, Literal

import pytorch_lightning as pl
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from embeddings.data.huggingface_datamodule import HuggingFaceDataset
from embeddings.model.model import Model
from embeddings.task.lightning_task.lightning_task import HuggingfaceLightningTask


class LightningModel(Model[pl.LightningDataModule, Dict[str, NDArray[Any]]]):
    def __init__(
        self,
        trainer: pl.Trainer,
        task: HuggingfaceLightningTask,
        predict_subset: Literal["dev", "test"] = "test",
    ) -> None:
        super().__init__()
        self.trainer = trainer
        self.task = task
        self.predict_subset = predict_subset

    def execute(self, data: pl.LightningDataModule) -> Dict[str, NDArray[Any]]:
        self.trainer.fit(self.task, data)
        self.trainer.test(datamodule=data)
        dataloader = (
            data.test_dataloader() if self.predict_subset == "test" else data.val_dataloader()
        )
        assert isinstance(dataloader, DataLoader)
        return self.task.predict(dataloader=dataloader)
