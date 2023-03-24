from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from embeddings.transformation.transformation import Transformation


class QuestionAnsweringSplitsTransformation(Transformation[Dataset, DatasetDict]):
    def __init__(
        self,
        train_size: float,
        dev_size: float,
        test_size: float,
        seed: int,
        context_column: str = "context",
        stratify_column: Optional[str] = None,
    ):
        self.train_size = train_size
        self.dev_size = dev_size
        self.test_size = test_size
        self.seed = seed
        self.context_column = context_column
        self.stratify_column = stratify_column

    def transform(self, data: Dataset) -> DatasetDict:
        data_df: pd.DataFrame = data.to_pandas()
        dataset = DatasetDict()
        unique_contexts = list(sorted(data_df[self.context_column].unique()))
        context_ids = list(range(len(unique_contexts)))
        contexts_mapping = dict(zip(unique_contexts, context_ids))
        data_df["context_id"] = data_df.apply(
            lambda x: contexts_mapping[x[self.context_column]], axis=1
        )

        stratify = None
        if self.stratify_column:
            assert self.stratify_column in data_df.columns
            stratify = []
            for context_id in context_ids:
                df = data_df[data_df.context_id == context_id]
                value = df[self.stratify_column].values
                assert len(value.unique()) == 1
                stratify.append(value.unique()[0])

        train_indices, validation_indices = train_test_split(
            context_ids,
            train_size=self.train_size,
            stratify=stratify,
            random_state=self.seed,
        )

        dataset["train"] = Dataset.from_pandas(
            data_df[data_df.context_id.isin(train_indices)], preserve_index=False
        )

        if self.dev_size and self.test_size:
            dev_indices, test_indices = train_test_split(
                validation_indices,
                train_size=round(self.dev_size / (1 - self.train_size), 2),
                random_state=self.seed,
            )

            dataset["validation"] = Dataset.from_pandas(
                data_df[data_df.context_id.isin(dev_indices)], preserve_index=False
            )
            dataset["test"] = Dataset.from_pandas(
                data_df[data_df.context_id.isin(test_indices)], preserve_index=False
            )

        elif self.dev_size and not self.test_size:
            dataset["validation"] = Dataset.from_pandas(
                data_df[data_df.context_id.isin(validation_indices)], preserve_index=False
            )

        elif self.test_size and not self.dev_size:
            dataset["test"] = Dataset.from_pandas(
                data_df[data_df.context_id.isin(validation_indices)], preserve_index=False
            )

        return dataset
