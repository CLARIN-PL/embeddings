import abc
from typing import Any, Dict, Generic, List, Optional, TypeVar

import numpy as np
from flair.data import Sentence
from numpy import typing as nptyping
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _VectorizerMixin

from embeddings.embedding.flair_embedding import FlairEmbedding
from embeddings.utils.array_like import ArrayLike

Output = TypeVar("Output")


# ignoring the mypy error due to no types (Any) in TransformerMixin and BaseEstimator classes
class FlairVectorizer(TransformerMixin, _VectorizerMixin, BaseEstimator, Generic[Output]):  # type: ignore
    def __init__(self, flair_embedding: FlairEmbedding) -> None:
        self.embedder = flair_embedding

    def fit(self, x: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        pass

    @abc.abstractmethod
    def transform(self, x: ArrayLike) -> Output:
        pass

    def fit_transform(self, x: ArrayLike, y: Optional[ArrayLike] = None, **kwargs: Any) -> Output:
        return self.transform(x)


class FlairDocumentVectorizer(FlairVectorizer[nptyping.NDArray[np.float_]]):
    def transform(self, x: ArrayLike) -> nptyping.NDArray[np.float_]:
        sentences = [Sentence(example) for example in x]
        embeddings = [sentence.embedding.numpy() for sentence in self.embedder.embed(sentences)]
        return np.vstack(embeddings)


class FlairWordVectorizer(FlairVectorizer[List[List[Dict[int, float]]]]):
    def transform(self, x: ArrayLike) -> List[List[Dict[int, float]]]:
        sentences = [Sentence(example) for example in x]
        embeddings = [sentence for sentence in self.embedder.embed(sentences)]
        return [
            [{i: value for i, value in enumerate(word.embedding.numpy())} for word in sent]
            for sent in embeddings
        ]
