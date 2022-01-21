from typing import Any, Dict, Optional

from numpy import typing as nptyping
from torch.utils.data import DataLoader

from embeddings.data.dataset import LightingDataLoaders
from embeddings.data.io import T_path
from embeddings.model.lightning_module.text_classification import TextClassificationModule
from embeddings.task.lightning_task.lightning_task import LightningTask


class TextClassificationTask(LightningTask):
    def __init__(
        self,
        model_name_or_path: str,
        output_path: T_path,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        finetune_last_n_layers: int = -1,
        model_config_kwargs: Optional[Dict[str, Any]] = None,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(output_path, task_train_kwargs)
        self.model_name_or_path = model_name_or_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.finetune_last_n_layers = finetune_last_n_layers
        self.model_config_kwargs = model_config_kwargs if model_config_kwargs else {}
        self.task_model_kwargs = task_model_kwargs if task_model_kwargs else {}

    def build_task_model(self) -> None:
        self.model = TextClassificationModule(
            model_name_or_path=self.model_name_or_path,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            finetune_last_n_layers=self.finetune_last_n_layers,
            config_kwargs=self.model_config_kwargs,
            task_model_kwargs=self.task_model_kwargs,
        )

    def predict(self, dataloader: DataLoader[Any]) -> Dict[str, nptyping.NDArray[Any]]:
        assert self.model is not None
        return self.model.predict(dataloader=dataloader)
