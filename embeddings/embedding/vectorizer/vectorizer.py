import abc
from typing import Any, Generic, Optional, TypeVar

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _VectorizerMixin

from embeddings.utils.array_like import ArrayLike

Output = TypeVar("Output")
Embedding = TypeVar("Embedding")


# ignoring the mypy error due to no types (Any) in TransformerMixin and BaseEstimator classes
class Vectorizer(
    TransformerMixin,  # type: ignore
    _VectorizerMixin,  # type: ignore
    BaseEstimator,  # type: ignore
    abc.ABC,
    Generic[Embedding, Output],
):
    def __init__(self, embedding: Embedding) -> None:
        self.embedding = embedding

    @abc.abstractmethod
    def fit(self, x: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        pass

    @abc.abstractmethod
    def transform(self, x: ArrayLike) -> Output:
        pass

    @abc.abstractmethod
    def fit_transform(self, x: ArrayLike, y: Optional[ArrayLike] = None, **kwargs: Any) -> Output:
        pass
