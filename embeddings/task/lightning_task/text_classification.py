from typing import Any, Dict, Optional

import numpy as np
from torch.utils.data import DataLoader

from embeddings.data.io import T_path
from embeddings.evaluator.evaluation_results import Predictions
from embeddings.model.lightning_module.text_classification import TextClassificationModule
from embeddings.task.lightning_task.lightning_task import LightningTask
from embeddings.utils.loggers import LightningLoggingConfig


class TextClassificationTask(LightningTask):
    def __init__(
        self,
        model_name_or_path: T_path,
        output_path: T_path,
        num_classes: int,
        model_config_kwargs: Dict[str, Any],
        task_model_kwargs: Dict[str, Any],
        task_train_kwargs: Dict[str, Any],
        early_stopping_kwargs: Dict[str, Any],
        model_checkpoint_kwargs: Dict[str, Any],
        logging_config: LightningLoggingConfig,
        finetune_last_n_layers: int = -1,
    ) -> None:
        super().__init__(
            output_path,
            task_train_kwargs,
            early_stopping_kwargs,
            model_checkpoint_kwargs,
            logging_config,
        )
        self.model_name_or_path = model_name_or_path
        self.num_classes = num_classes
        self.model_config_kwargs = model_config_kwargs
        self.task_model_kwargs = task_model_kwargs
        self.finetune_last_n_layers = finetune_last_n_layers
        self.task_train_kwargs = task_train_kwargs

    def build_task_model(self) -> None:
        self.model = TextClassificationModule(
            model_name_or_path=self.model_name_or_path,
            num_classes=self.num_classes,
            finetune_last_n_layers=self.finetune_last_n_layers,
            config_kwargs=self.model_config_kwargs,
            task_model_kwargs=self.task_model_kwargs,
        )

    def predict(self, dataloader: DataLoader[Any], return_names: bool = True) -> Predictions:
        assert self.model is not None
        results = self.model.predict(dataloader=dataloader)
        results["names"] = np.array(self.model.target_names)
        return Predictions(**results)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: T_path,
        output_path: T_path,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
        logging_config: Optional[LightningLoggingConfig] = None,
    ) -> "LightningTask":
        return cls.restore_task_model(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            task_train_kwargs=task_train_kwargs,
            lightning_module=TextClassificationModule,
            logging_config=logging_config,
        )
