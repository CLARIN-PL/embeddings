from typing import Any, Dict

from numpy import typing as nptyping
from torch.utils.data import DataLoader

from embeddings.data.io import T_path
from embeddings.model.lightning_module.text_classification import TextClassificationModule
from embeddings.task.lightning_task.lightning_task import LightningTask


class TextClassificationTask(LightningTask):
    def __init__(
        self,
        model_name_or_path: T_path,
        output_path: T_path,
        model_config_kwargs: Dict[str, Any],
        task_model_kwargs: Dict[str, Any],
        task_train_kwargs: Dict[str, Any],
        early_stopping_kwargs: Dict[str, Any],
        finetune_last_n_layers: int = -1,
    ) -> None:
        super().__init__(output_path, task_train_kwargs, early_stopping_kwargs)
        self.model_name_or_path = model_name_or_path
        self.model_config_kwargs = model_config_kwargs
        self.task_model_kwargs = task_model_kwargs
        self.finetune_last_n_layers = finetune_last_n_layers

    def build_task_model(self) -> None:
        self.model = TextClassificationModule(
            model_name_or_path=self.model_name_or_path,
            finetune_last_n_layers=self.finetune_last_n_layers,
            config_kwargs=self.model_config_kwargs,
            task_model_kwargs=self.task_model_kwargs,
        )

    def predict(self, dataloader: DataLoader[Any]) -> Dict[str, nptyping.NDArray[Any]]:
        assert self.model is not None
        return self.model.predict(dataloader=dataloader)
