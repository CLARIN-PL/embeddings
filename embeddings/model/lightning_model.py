from typing import Any, Dict

from numpy import typing as nptyping

from embeddings.data.datamodule import HuggingFaceDataModule
from embeddings.data.dataset import LightingDataModuleSubset
from embeddings.model.model import Model
from embeddings.task.lightning_task.lightning_task import LightningTask


class LightningModel(Model[HuggingFaceDataModule, Dict[str, nptyping.NDArray[Any]]]):
    def __init__(
        self,
        task: LightningTask,
        predict_subset: LightingDataModuleSubset,
    ) -> None:
        super().__init__()
        self.task = task
        self.predict_subset = predict_subset

    def execute(self, data: HuggingFaceDataModule) -> Dict[str, nptyping.NDArray[Any]]:
        self.task.build_task_model()
        return self.task.fit_predict(data, self.predict_subset)
