import abc
from pathlib import Path
from typing import Dict, Tuple, Any, List, Union
from typing import Optional

import flair
import numpy as np
import torch
from datasets import Metric
from flair.data import Corpus
from flair.trainers import ModelTrainer

from experimental.data.io import T_path


class BaseTask(abc.ABC):
    def __init__(
        self,
        model: flair.nn.Model,
        output_path: T_path,
        metrics: Optional[List[Tuple[Metric, Dict[str, Any]]]] = None,
    ):
        self.output_path: Path = Path(output_path)
        self.model: flair.nn.Model = model
        self.trainer: Optional[ModelTrainer] = None

        if metrics is None:
            metrics = self._get_default_metrics()

        self.metrics = metrics

    def fit(
        self,
        corpus: Corpus,
        learning_rate: float = 0.1,
        mini_batch_size: int = 32,
        max_epochs: int = 1,
    ) -> Dict[Any, Any]:
        self.trainer = ModelTrainer(self.model, corpus)
        log: Dict[Any, Any] = self.trainer.train(
            base_path=self.output_path,
            learning_rate=learning_rate,
            mini_batch_size=mini_batch_size,
            max_epochs=max_epochs,
        )
        return log

    @abc.abstractmethod
    def _get_default_metrics(self) -> List[Tuple[Metric, Dict[str, Any]]]:
        pass

    def compute_metrics(
        self,
        y_true: Optional[Union[List[Any], np.ndarray, torch.Tensor]],
        y_pred: Optional[Union[List[Any], np.ndarray, torch.Tensor]],
    ) -> List[Optional[Dict[Any, Any]]]:
        return [
            metric.compute(references=y_true, predictions=y_pred, **kwargs)
            for metric, kwargs in self.metrics
        ]
