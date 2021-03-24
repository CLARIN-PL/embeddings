import abc
import os
from typing import Dict

from flair.data import Dictionary
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.training_utils import Result


class BaseSequenceTagger:
    @abc.abstractmethod
    def fit(self, dataset) -> None:
        pass

    def evaluate(self, dataset):
        pass


class FlairSequenceTagger:
    def __init__(
        self,
        embeddings,
        hidden_dim: float,
        tag_dictionary: Dictionary,
        output_path: str,
        use_crf: bool = True,
    ):
        self.model_hparams = {
            "hidden_size": hidden_dim,
            "use_crf": use_crf,
        }
        self.model = SequenceTagger(
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type="tag",
            **self.model_hparams
        )
        self.output_path = output_path
        self.trainer = None

    def fit(
        self,
        corpus: ColumnCorpus,
        learning_rate: float = 0.1,
        mini_batch_size: int = 32,
        max_epochs: int = 1,
    ) -> Dict:
        self.trainer = ModelTrainer(self.model, corpus)
        log = self.trainer.train(
            os.path.join(self.output_path, "tagger/"),
            learning_rate=learning_rate,
            mini_batch_size=mini_batch_size,
            max_epochs=max_epochs,
        )
        return log

    def evaluate(self, corpus) -> (Result, float):
        return self.model.evaluate(corpus.test)
