from typing import Any, Generic, Optional

from embeddings.data.dataset import LightingDataModuleSubset
from embeddings.model.model import Model
from embeddings.task.lightning_task.lightning_task import LightningDataModule, LightningTask
from embeddings.task.task import Output


class LightningModel(
    Model[LightningDataModule, Output],
    Generic[LightningDataModule, Output],
):
    def __init__(
        self,
        task: LightningTask[LightningDataModule, Output],
        predict_subset: LightingDataModuleSubset,
    ) -> None:
        super().__init__()
        self.task = task
        self.predict_subset = predict_subset

    def execute(
        self, data: LightningDataModule, run_name: Optional[str] = None, **kwargs: Any
    ) -> Output:
        self.task.build_task_model()
        return self.task.fit_predict(data, self.predict_subset, run_name=run_name)
