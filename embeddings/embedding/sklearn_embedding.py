from typing import Any, Dict, Optional

import pandas as pd
from sklearn.base import BaseEstimator as AnySklearnVectorizer

from embeddings.embedding.embedding import Embedding
from embeddings.utils.array_like import ArrayLike


class SklearnEmbedding(Embedding[ArrayLike, pd.DataFrame]):
    def __init__(
        self, vectorizer: AnySklearnVectorizer, embedding_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.embedding_kwargs = embedding_kwargs if embedding_kwargs else {}
        self.vectorizer = vectorizer(**self.embedding_kwargs)

    def fit(self, data: ArrayLike) -> None:
        self.vectorizer.fit(data)

    def embed(self, data: ArrayLike) -> pd.DataFrame:
        return pd.DataFrame(
            self.vectorizer.transform(data).A, columns=self.vectorizer.get_feature_names_out()
        )
