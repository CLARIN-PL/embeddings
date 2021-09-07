import abc
from abc import ABC
from typing import Any, List

from flair.data import Sentence
from flair.embeddings import (
    DocumentCNNEmbeddings,
    DocumentPoolEmbeddings,
    DocumentRNNEmbeddings,
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


class FlairAggregationEmbedding(FlairEmbedding, ABC):
    def __init__(self, name: str, word_embedding: FlairEmbedding, **load_model_kwargs: Any):
        if not isinstance(word_embedding.model, TokenEmbeddings):
            raise ValueError(
                f"{object.__repr__(word_embedding.model)} is not an instance of {TokenEmbeddings}."
            )
        self.word_embedding = word_embedding
        super().__init__(name, **load_model_kwargs)


class FlairDocumentPoolEmbedding(FlairAggregationEmbedding):
    def __init__(
        self,
        word_embedding: FlairEmbedding,
        pooling: str = "mean",
        **load_model_kwargs: Any,
    ):
        self.pooling = pooling
        super().__init__(
            f"{pooling}-pooling({word_embedding.name})", word_embedding, **load_model_kwargs
        )

    def _get_model(self) -> Embeddings:
        return DocumentPoolEmbeddings(
            [self.word_embedding.model], pooling=self.pooling, **self.load_model_kwargs
        )


class FlairDocumentRNNEmbeddings(FlairAggregationEmbedding):
    def __init__(self, word_embedding: FlairEmbedding, **load_model_kwargs: Any):
        super().__init__(f"RNN({word_embedding.name})", word_embedding, **load_model_kwargs)

    def _get_model(self) -> Embeddings:
        return DocumentRNNEmbeddings([self.word_embedding.model], **self.load_model_kwargs)


class FlairDocumentCNNEmbeddings(FlairAggregationEmbedding):
    def __init__(self, word_embedding: FlairEmbedding, **load_model_kwargs: Any):
        super().__init__(f"CNN({word_embedding.name})", word_embedding, **load_model_kwargs)

    def _get_model(self) -> Embeddings:
        return DocumentCNNEmbeddings([self.word_embedding.model], **self.load_model_kwargs)
