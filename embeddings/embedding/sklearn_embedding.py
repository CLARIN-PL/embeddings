from typing import Any, Dict, Optional

import pandas as pd
import scipy
from sklearn.base import BaseEstimator as AnySklearnVectorizer

from embeddings.embedding.embedding import Embedding
from embeddings.utils.array_like import ArrayLike


class SklearnEmbedding(Embedding[ArrayLike, pd.DataFrame]):
    def __init__(
        self, vectorizer: AnySklearnVectorizer, vectorizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.vectorizer = vectorizer(**vectorizer_kwargs if vectorizer_kwargs else {})

    def fit(self, data: ArrayLike) -> None:
        self.vectorizer.fit(data)

    def embed(self, data: ArrayLike) -> pd.DataFrame:
        embedded = self.vectorizer.transform(data)
        if scipy.sparse.issparse(embedded):
            embedded = embedded.A
        return pd.DataFrame(embedded)
