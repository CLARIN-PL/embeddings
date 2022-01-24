from typing import Dict, Any

from numpy import typing as nptyping
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from typing_extensions import Literal

from embeddings.model.model import Model
from embeddings.task.sklearn_task.text_classification import TextClassification


class SklearnModel(Model[Any[pd.DataFrame, nptyping.NDArray[Any]], Dict]):
    def __init__(
        self, model: Any[DecisionTreeClassifier, RandomForestClassifier, LogisticRegression]
    ):
        super().__init__()
        self.model = model

    def execute(self, data):
        pass
