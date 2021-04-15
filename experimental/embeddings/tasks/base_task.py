import abc
from pathlib import Path
from typing import Dict, Tuple, Any
from typing import Optional

import flair
from flair.data import FlairDataset, Corpus
from flair.trainers import ModelTrainer
from flair.training_utils import Result


class BaseTask(abc.ABC):
    def __init__(
        self,
        model: flair.nn.Model,
        output_path: str,
    ):
        self.output_path: Path = Path(output_path)
        self.model: flair.nn.Model = model
        self.trainer: Optional[ModelTrainer] = None

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

    def evaluate(self, dataset: FlairDataset) -> Tuple[Result, float]:
        log: Tuple[Result, float] = self.model.evaluate(dataset)
        return log
