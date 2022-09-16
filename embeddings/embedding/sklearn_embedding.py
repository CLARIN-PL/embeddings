from typing import Any, Dict, Optional

import pandas as pd
from sklearn.base import BaseEstimator as AnySklearnVectorizer

from embeddings.embedding.embedding import Embedding
from embeddings.utils.array_like import ArrayLike


class SklearnEmbedding(Embedding[ArrayLike, pd.DataFrame]):
    def __init__(
        self,
        vectorizer: AnySklearnVectorizer,
        vectorizer_has_sparse_output: bool = True,
        vectorizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.vectorizer_kwargs = vectorizer_kwargs if vectorizer_kwargs else {}
        self.vectorizer_has_sparse_output = vectorizer_has_sparse_output
        self.vectorizer = vectorizer(**self.vectorizer_kwargs)

    def fit(self, data: ArrayLike) -> None:
        self.vectorizer.fit(data)

    def embed(self, data: ArrayLike) -> pd.DataFrame:
        embedded = self.vectorizer.transform(data)
        if self.vectorizer_has_sparse_output:
            embedded = embedded.A
        return pd.DataFrame(embedded, columns=self.vectorizer.get_feature_names_out())
