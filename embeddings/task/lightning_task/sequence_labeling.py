from typing import Any, Dict, List, Optional

import numpy as np
from numpy import typing as nptyping
from torch.utils.data import DataLoader

from embeddings.data.io import T_path
from embeddings.model.lightning_module.sequence_labeling import SequenceLabelingModule
from embeddings.task.lightning_task.lightning_task import LightningTask


class SequenceLabelingTask(LightningTask):
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
        self.model = SequenceLabelingModule(
            model_name_or_path=self.model_name_or_path,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            finetune_last_n_layers=self.finetune_last_n_layers,
            config_kwargs=self.model_config_kwargs,
            task_model_kwargs=self.task_model_kwargs,
        )

    def predict(self, dataloader: DataLoader[Any]) -> Dict[str, nptyping.NDArray[Any]]:
        assert self.model is not None
        predictions, ground_truth = map(
            np.ndarray.tolist, self.model.predict(dataloader=dataloader).values()
        )
        for i, (pred, gt) in enumerate(zip(list(predictions), list(ground_truth))):
            predictions[i] = self._map_filter_data(pred, gt)
            ground_truth[i] = self._map_filter_data(gt, gt)
        return {"y_pred": np.array(predictions), "y_true": np.array(ground_truth)}

    def _map_filter_data(
        self, data: nptyping.NDArray[Any], ground_truth_data: nptyping.NDArray[Any]
    ) -> List[str]:
        assert self.model is not None
        assert self.trainer is not None
        assert hasattr(self.trainer, "datamodule")
        return [
            getattr(self.trainer, "datamodule").id2str(x.item())
            for x in data[ground_truth_data != self.model.ignore_index]
        ]
