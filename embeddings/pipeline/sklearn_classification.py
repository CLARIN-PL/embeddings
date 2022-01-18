from typing import Tuple, Dict

import datasets
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class SklearnClassificationPipeline:
    def __init__(
        self, dataset_name: str, target_name: str, feature_name: str, max_features: int = 50
    ):

        self.dataset_name = dataset_name
        self.target_name = target_name
        self.feature_name = feature_name
        self.max_features = max_features
        self.df_train = pd.DataFrame()
        self.df_valid = pd.DataFrame()
        self.df_test = pd.DataFrame()
        self._prepare_data()
        self.random_forest = None
        self.tree = None
        self.logistic = None
        self.vectorizer = None

    def _prepare_data(self) -> None:

        dataset = datasets.load_dataset(self.dataset_name)

        for row in dataset["train"]:
            self.df_train = self.df_train.append(row, ignore_index=True)

        for row in dataset["validation"]:
            self.df_valid = self.df_valid.append(row, ignore_index=True)

        for row in dataset["test"]:
            self.df_test = self.df_test.append(row, ignore_index=True)

    def vectorize_text(
        self, vectorization_method: str = "tfidf"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if vectorization_method == "bow":
            self.vectorizer = CountVectorizer(max_features=self.max_features)
        elif vectorization_method == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=self.max_features)

        self.vectorizer.fit(self.df_train[self.feature_name])

        train_data = self.vectorizer.transform(self.df_train[self.feature_name]).toarray()
        valid_data = self.vectorizer.transform(self.df_valid[self.feature_name]).toarray()
        test_data = self.vectorizer.transform(self.df_test[self.feature_name]).toarray()

        return train_data, valid_data, test_data

    def train_decision_tree(self, train_data):

        target = self.df_train[self.target_name].values
        tree = DecisionTreeClassifier()
        tree.fit(train_data, target)
        self.tree = tree

    def train_logistic(self, train_data):

        target = self.df_train[self.target_name].values
        logistic = LogisticRegression()
        logistic.fit(train_data, target)
        self.logistic = logistic

    def train_random_forest(self, train_data):

        target = self.df_train[self.target_name].values
        random_forest = RandomForestClassifier()
        random_forest.fit(train_data, target)
        self.random_forest = random_forest

    def evaluate_models(
        self, train_data, valid_data, test_data
    ) -> Dict[str, Dict[str, Dict[str, float]]]:

        target_train = self.df_train[self.target_name].values
        target_valid = self.df_valid[self.target_name].values
        target_test = self.df_test[self.target_name].values

        tree_preds_train = self.tree.predict(train_data)
        tree_preds_valid = self.tree.predict(valid_data)
        tree_preds_test = self.tree.predict(test_data)

        logistic_preds_train = self.logistic.predict(train_data)
        logistic_preds_valid = self.logistic.predict(valid_data)
        logistic_preds_test = self.logistic.predict(test_data)

        random_forest_preds_train = self.random_forest.predict(train_data)
        random_forest_preds_valid = self.random_forest.predict(valid_data)
        random_forest_preds_test = self.random_forest.predict(test_data)

        return {
            "decision_tree": {
                "accuracy": {
                    "train": accuracy_score(target_train, tree_preds_train),
                    "valid": accuracy_score(target_valid, tree_preds_valid),
                    "test": accuracy_score(target_test, tree_preds_test),
                },
                "precision": {
                    "train": precision_score(target_train, tree_preds_train, average="macro"),
                    "valid": precision_score(target_valid, tree_preds_valid, average="macro"),
                    "test": precision_score(target_test, tree_preds_test, average="macro"),
                },
                "recall": {
                    "train": recall_score(target_train, tree_preds_train, average="macro"),
                    "valid": recall_score(target_valid, tree_preds_valid, average="macro"),
                    "test": recall_score(target_test, tree_preds_test, average="macro"),
                },
                "f1": {
                    "train": f1_score(target_train, tree_preds_train, average="macro"),
                    "valid": f1_score(target_valid, tree_preds_valid, average="macro"),
                    "test": f1_score(target_test, tree_preds_test, average="macro"),
                },
            },
            "logistic_regression": {
                "accuracy": {
                    "train": accuracy_score(target_train, logistic_preds_train),
                    "valid": accuracy_score(target_valid, logistic_preds_valid),
                    "test": accuracy_score(target_test, logistic_preds_test),
                },
                "precision": {
                    "train": precision_score(target_train, logistic_preds_train, average="macro"),
                    "valid": precision_score(target_valid, logistic_preds_valid, average="macro"),
                    "test": precision_score(target_test, logistic_preds_test, average="macro"),
                },
                "recall": {
                    "train": recall_score(target_train, logistic_preds_train, average="macro"),
                    "valid": recall_score(target_valid, logistic_preds_valid, average="macro"),
                    "test": recall_score(target_test, logistic_preds_test, average="macro"),
                },
                "f1": {
                    "train": f1_score(target_train, logistic_preds_train, average="macro"),
                    "valid": f1_score(target_valid, logistic_preds_valid, average="macro"),
                    "test": f1_score(target_test, logistic_preds_test, average="macro"),
                },
            },
            "random_forest": {
                "accuracy": {
                    "train": accuracy_score(target_train, random_forest_preds_train),
                    "valid": accuracy_score(target_valid, random_forest_preds_valid),
                    "test": accuracy_score(target_test, random_forest_preds_test),
                },
                "precision": {
                    "train": precision_score(target_train, random_forest_preds_train, average="macro"),
                    "valid": precision_score(target_valid, random_forest_preds_valid, average="macro"),
                    "test": precision_score(target_test, random_forest_preds_test, average="macro"),
                },
                "recall": {
                    "train": recall_score(target_train, random_forest_preds_train, average="macro"),
                    "valid": recall_score(target_valid, random_forest_preds_valid, average="macro"),
                    "test": recall_score(target_test, random_forest_preds_test, average="macro"),
                },
                "f1": {
                    "train": f1_score(target_train, random_forest_preds_train, average="macro"),
                    "valid": f1_score(target_valid, random_forest_preds_valid, average="macro"),
                    "test": f1_score(target_test, random_forest_preds_test, average="macro"),
                },
            },
        }

    def run(self, vectorization_method: str = "tfidf") -> Dict[str, Dict[str, Dict[str, float]]]:

        print("## Vectorizing text")
        train_data, valid_data, test_data = self.vectorize_text(
            vectorization_method=vectorization_method
        )
        print("## Done")
        print("## Training logistic regression")
        self.train_logistic(train_data)
        print("## Done")
        print("## Training decision tree")
        self.train_decision_tree(train_data)
        print("## Done")
        print("## Training random_forest")
        self.train_random_forest(train_data)
        print("## Done")
        print("## Evaluating")
        pipeline_metrics = self.evaluate_models(train_data, valid_data, test_data)
        print("## Pipeline finished")
        return pipeline_metrics
