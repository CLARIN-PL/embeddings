from typing import List, Tuple, Any

import datasets
from experimental.embeddings.tasks.base_task import BaseTask
from flair.data import Dictionary, Sentence
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger


def get_labels_from_flair_sentences(
    sentences: List[Sentence],
) -> Tuple[List[List[str]], List[List[str]]]:
    y_true = []
    y_pred = []

    for sent in sentences:
        sent_true = []
        sent_pred = []
        for token in sent:
            sent_true.append(token.get_tag("tag").value)
            sent_pred.append(token.get_tag("prediction").value)

        y_true.append(sent_true)
        y_pred.append(sent_pred)

    return y_true, y_pred


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

    def evaluate(self, sentences: List[Sentence]) -> Tuple[Any, float]:
        self.model.predict(sentences, label_name="prediction")
        y_true, y_pred = get_labels_from_flair_sentences(sentences)

        seqeval = datasets.load_metric("seqeval")
        results = seqeval.compute(
            predictions=y_pred, references=y_true, suffix=None, scheme="BILOU", mode="strict"
        )

        return results
