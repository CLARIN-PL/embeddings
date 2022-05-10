from typing import Any, Optional

from embeddings.data.datamodule import HuggingFaceDataModule
from embeddings.data.dataset import LightingDataModuleSubset
from embeddings.evaluator.evaluation_results import Predictions
from embeddings.model.model import Model
from embeddings.task.lightning_task.lightning_task import LightningTask


class LightningModel(Model[HuggingFaceDataModule, Predictions]):
    def __init__(
        self,
        task: LightningTask,
        predict_subset: LightingDataModuleSubset,
    ) -> None:
        super().__init__()
        self.task = task
        self.predict_subset = predict_subset

    def execute(
        self, data: HuggingFaceDataModule, run_name: Optional[str] = None, **kwargs: Any
    ) -> Predictions:
        self.task.build_task_model()
        return self.task.fit_predict(data, self.predict_subset, run_name=run_name)
