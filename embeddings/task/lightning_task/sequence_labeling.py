from typing import Any, Dict, List, Optional

import numpy as np
from datasets import ClassLabel
from numpy import typing as nptyping
from torch.utils.data import DataLoader

from embeddings.data.io import T_path
from embeddings.evaluator.evaluation_results import Predictions
from embeddings.metric.sequence_labeling import EvaluationMode, TaggingScheme
from embeddings.model.lightning_module.sequence_labeling import SequenceLabelingModule
from embeddings.task.lightning_task.lightning_task import LightningTask
from embeddings.utils.loggers import LightningLoggingConfig


class SequenceLabelingTask(LightningTask):
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
        evaluation_mode: EvaluationMode = EvaluationMode.CONLL,
        tagging_scheme: Optional[TaggingScheme] = None,
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
        self.evaluation_mode = evaluation_mode
        self.tagging_scheme = tagging_scheme

    def build_task_model(self) -> None:
        self.model = SequenceLabelingModule(
            model_name_or_path=self.model_name_or_path,
            num_classes=self.num_classes,
            finetune_last_n_layers=self.finetune_last_n_layers,
            evaluation_mode=self.evaluation_mode,
            tagging_scheme=self.tagging_scheme,
            config_kwargs=self.model_config_kwargs,
            task_model_kwargs=self.task_model_kwargs,
        )

    def predict(self, dataloader: DataLoader[Any], return_names: bool = True) -> Predictions:
        assert self.model is not None
        results = self.model.predict(dataloader=dataloader)
        predictions, ground_truth, probabilities = (
            list(results["y_pred"]),
            list(results["y_true"]),
            list(results["y_probabilities"]),
        )

        for i, (pred, gt, probs) in enumerate(zip(predictions, ground_truth, probabilities)):
            predictions[i] = self._map_filter_data(pred, gt)
            ground_truth[i] = self._map_filter_data(gt, gt)
            probabilities[i] = [x for x in probs[gt != self.model.ignore_index]]

        results = {
            "y_pred": np.array(predictions, dtype=object),
            "y_true": np.array(ground_truth, dtype=object),
            "y_probabilities": np.array(probabilities, dtype=object),
            "names": np.array(self.model.target_names),
        }
        return Predictions(**results)

    def _map_filter_data(
        self, data: nptyping.NDArray[Any], ground_truth_data: nptyping.NDArray[Any]
    ) -> List[str]:
        assert self.model is not None
        assert isinstance(self.model.class_label, ClassLabel)
        return [
            str(self.model.class_label.int2str(x.item()))
            for x in data[ground_truth_data != self.model.ignore_index]
        ]

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
            lightning_module=SequenceLabelingModule,
            logging_config=logging_config,
        )
