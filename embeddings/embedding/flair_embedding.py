import abc
from typing import Any, List

from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings, TransformerWordEmbeddings
from flair.embeddings.base import Embeddings

from embeddings.embedding.embedding import Embedding


class FlairEmbedding(Embedding[List[Sentence], List[Sentence]]):
    def __init__(self, name: str, fine_tune: bool = False, **load_model_kwargs: Any) -> None:
        super().__init__()
        self.load_model_kwargs = load_model_kwargs
        self.load_model_kwargs["fine_tune"] = fine_tune
        self.name = name
        self.model = self._get_model()

    def embed(self, data: List[Sentence]) -> List[Sentence]:
        self.model.embed(data)
        return data

    @abc.abstractmethod
    def _get_model(self) -> Embeddings:
        pass


class FlairTransformerWordEmbedding(FlairEmbedding):
    def _get_model(self) -> TransformerWordEmbeddings:
        return TransformerWordEmbeddings(self.name, **self.load_model_kwargs)


class FlairTransformerDocumentEmbedding(FlairEmbedding):
    def _get_model(self) -> TransformerDocumentEmbeddings:
        return TransformerDocumentEmbeddings(self.name, **self.load_model_kwargs)
