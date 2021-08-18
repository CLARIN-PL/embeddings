import abc
from abc import ABC
from typing import Any, List, Union

from flair.data import Sentence
from flair.embeddings import (
    DocumentPoolEmbeddings,
    TokenEmbeddings,
    TransformerDocumentEmbeddings,
    TransformerWordEmbeddings,
)
from flair.embeddings.base import Embeddings

from embeddings.embedding.embedding import Embedding


class FlairEmbedding(Embedding[List[Sentence], List[Sentence]]):
    def __init__(self, name: str, **load_model_kwargs: Any) -> None:
        super().__init__()
        self.load_model_kwargs = load_model_kwargs
        self.name = name
        self.model = self._get_model()

    def embed(self, data: List[Sentence]) -> List[Sentence]:
        self.model.embed(data)
        return data

    @abc.abstractmethod
    def _get_model(self) -> Embeddings:
        pass


class FlairTransformerEmbedding(FlairEmbedding, ABC):
    def __init__(self, name: str, fine_tune: bool = False, **load_model_kwargs: Any) -> None:
        load_model_kwargs["fine_tune"] = fine_tune
        super().__init__(name, **load_model_kwargs)


class FlairTransformerWordEmbedding(FlairTransformerEmbedding):
    def _get_model(self) -> TransformerWordEmbeddings:
        return TransformerWordEmbeddings(self.name, **self.load_model_kwargs)


class FlairTransformerDocumentEmbedding(FlairTransformerEmbedding):
    def _get_model(self) -> TransformerDocumentEmbeddings:
        return TransformerDocumentEmbeddings(self.name, **self.load_model_kwargs)


class FlairDocumentPoolEmbedding(Embedding[List[Sentence], List[Sentence]]):
    def __init__(
        self,
        embeddings: Union[FlairEmbedding, List[FlairEmbedding]],
        pooling: str = "mean",
        **kwargs: Any,
    ):
        super().__init__()
        embeddings = [embeddings] if isinstance(embeddings, FlairEmbedding) else embeddings
        embeddings = [e.model for e in embeddings]
        for e in embeddings:
            if not isinstance(e, TokenEmbeddings):
                raise ValueError(f"{e} is not an instance of {TokenEmbeddings}.")
        self.model = DocumentPoolEmbeddings(embeddings, pooling=pooling, **kwargs)

    def embed(self, data: List[Sentence]) -> List[Sentence]:
        self.model.embed(data)
        return data
