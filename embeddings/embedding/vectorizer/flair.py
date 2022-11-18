import abc
from typing import Any, Dict, Generic, List, Optional

import numpy as np
from flair.data import Sentence
from numpy import typing as nptyping

from embeddings.embedding.flair_embedding import FlairEmbedding
from embeddings.embedding.vectorizer.vectorizer import Output, Vectorizer
from embeddings.utils.array_like import ArrayLike


class FlairVectorizer(Vectorizer[FlairEmbedding, Output], abc.ABC, Generic[Output]):
    def fit(self, x: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        pass

    def fit_transform(self, x: ArrayLike, y: Optional[ArrayLike] = None, **kwargs: Any) -> Output:
        return self.transform(x)


class FlairDocumentVectorizer(FlairVectorizer[nptyping.NDArray[np.float_]]):
    def transform(self, x: ArrayLike) -> nptyping.NDArray[np.float_]:
        sentences = [Sentence(example) for example in x]
        embeddings = [sentence.embedding.numpy() for sentence in self.embedding.embed(sentences)]
        return np.vstack(embeddings)


class FlairWordVectorizer(FlairVectorizer[List[List[Dict[int, float]]]]):
    def transform(self, x: ArrayLike) -> List[List[Dict[int, float]]]:
        sentences = [Sentence(example) for example in x]
        embeddings = [sentence for sentence in self.embedding.embed(sentences)]
        return [
            [{i: value for i, value in enumerate(word.embedding.numpy())} for word in sent]
            for sent in embeddings
        ]
