from typing import Any, Dict, Union

from numpy import typing as nptyping
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from embeddings.embedding.embedding import Embedding


class SklearnEmbedding(Embedding[Union[pd.Series, nptyping.NDArray[Any]], pd.DataFrame]):
    def __init__(self, embedding_kwargs: Dict[str, any], method: str):
        super().__init__()
        self.embedding_kwargs = embedding_kwargs if embedding_kwargs else {}
        self.method = method

    def fit(self, data: Union[pd.Series, nptyping.NDArray[Any]]) -> None:
        self.vectorizer = (
            CountVectorizer(**self.embedding_kwargs)
            if self.method == "bow"
            else TfidfVectorizer(**self.embedding_kwargs)
        )
        self.vectorizer.fit(data)

    def embed(self, data: Union[pd.Series, nptyping.NDArray[Any]]) -> pd.DataFrame:
        return pd.DataFrame(
            self.vectorizer.transform(data).A, columns=self.vectorizer.get_feature_names_out()
        )
