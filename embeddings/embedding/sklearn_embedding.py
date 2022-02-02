from typing import Any, Dict, Union

import pandas as pd
from numpy import typing as nptyping
from sklearn.base import BaseEstimator as AnySklearnVectorizer

from embeddings.embedding.embedding import Embedding


class SklearnEmbedding(Embedding[Union[pd.Series, nptyping.NDArray[Any]], pd.DataFrame]):
    def __init__(self, embedding_kwargs: Dict[str, Any], vectorizer: AnySklearnVectorizer):
        super().__init__()
        self.embedding_kwargs = embedding_kwargs if embedding_kwargs else {}
        self.vectorizer = vectorizer(**self.embedding_kwargs)

    def fit(self, data: Union[pd.Series, nptyping.NDArray[Any]]) -> None:
        self.vectorizer.fit(data)

    def embed(self, data: Union[pd.Series, nptyping.NDArray[Any]]) -> pd.DataFrame:
        return pd.DataFrame(
            self.vectorizer.transform(data).A, columns=self.vectorizer.get_feature_names_out()
        )
