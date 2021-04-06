from pathlib import Path
from typing import Any, Optional

from flair.data import Dictionary
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


class FlairSequenceTagger:
    def __init__(
        self,
        embeddings: TransformerWordEmbeddings,
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
        self.output_path = Path(output_path)
        self.trainer: Optional[ModelTrainer] = None

    def fit(
        self,
        corpus: ColumnCorpus,
        learning_rate: float = 0.1,
        mini_batch_size: int = 32,
        max_epochs: int = 1,
    ) -> Any:
        self.trainer = ModelTrainer(self.model, corpus)
        log = self.trainer.train(
            base_path=self.output_path,
            learning_rate=learning_rate,
            mini_batch_size=mini_batch_size,
            max_epochs=max_epochs,
        )
        return log

    def evaluate(self, corpus: ColumnCorpus) -> Any:
        return self.model.evaluate(corpus.test)
