from typing import Any, Dict, Optional, TypeVar, Union

import pandas as pd
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _VectorizerMixin

from embeddings.embedding.embedding import Embedding
from embeddings.embedding.vectorizer.vectorizer import Vectorizer
from embeddings.utils.array_like import ArrayLike

SklearnVectorizer = TypeVar(
    "SklearnVectorizer", bound=Union[Vectorizer, _VectorizerMixin, BaseEstimator]
)


class SklearnEmbedding(Embedding[ArrayLike, pd.DataFrame]):
    def __init__(
        self, vectorizer: SklearnVectorizer, vectorizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        assert callable(vectorizer)
        self.vectorizer_kwargs = vectorizer_kwargs if vectorizer_kwargs else {}
        self.vectorizer = vectorizer(**self.vectorizer_kwargs)

    def fit(self, data: ArrayLike) -> None:
        self.vectorizer.fit(data)

    def embed(self, data: ArrayLike) -> pd.DataFrame:
        embedded = self.vectorizer.transform(data)
        if scipy.sparse.issparse(embedded):
            embedded = embedded.A
        return pd.DataFrame(embedded)
