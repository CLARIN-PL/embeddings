from typing import List, Tuple, Dict, Any

from datasets import Metric
from flair.data import Dictionary
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger

from experimental.embeddings.tasks.base_task import BaseTask


class FlairSequenceTagger(BaseTask):
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
        model = SequenceTagger(
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type="tag",
            **self.model_hparams
        )
        super().__init__(model, output_path)

    def _get_default_metrics(self) -> List[Tuple[Metric, Dict[str, Any]]]:
        raise NotImplementedError
