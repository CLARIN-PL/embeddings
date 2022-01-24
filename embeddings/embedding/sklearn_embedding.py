from typing import Any, Dict, Optional, Union

from numpy import typing as nptyping
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from embeddings.embedding.embedding import Embedding


class SklearnEmbedding(Embedding[Union[pd.Series, nptyping.NDArray[Any]], pd.DataFrame]):
    def __init__(self, method: str = "bow"):
        super().__init__()
        self.method = method

    def fit(
        self,
        data: Union[pd.Series, nptyping.NDArray[Any]],
        fit_vectorizer_kwargs: Optional[Dict[str, any]] = None,
    ) -> None:
        fit_vectorizer_kwargs = fit_vectorizer_kwargs if fit_vectorizer_kwargs else {}
        self.vectorizer = (
            CountVectorizer(**fit_vectorizer_kwargs)
            if self.method == "bow"
            else TfidfVectorizer(**fit_vectorizer_kwargs)
        )
        self.vectorizer.fit(data)

    def embed(self, data: Union[pd.Series, nptyping.NDArray[Any]]) -> pd.DataFrame:
        return pd.DataFrame(
            self.vectorizer.transform(data).A, columns=self.vectorizer.get_feature_names_out()
        )
