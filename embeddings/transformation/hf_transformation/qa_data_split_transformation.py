from typing import Optional

import pandas as pd
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split


def split_train_dev_test(
    ds: pd.DataFrame,
    train_size: float,
    dev_size: float,
    test_size: float,
    seed: int,
    context_column: str = "context",
    stratify_column: Optional[str] = None,
) -> DatasetDict:
    """
    TODO: Move to embeddings/transformation/hf_transformation/qa_data_split_transformation.py
    Refactor pt 2
    Q&A require separate split_train pipeline refactor it as a transformation and improve parametrization"
    """
    dataset = DatasetDict()
    unique_contexts = list(sorted(ds[context_column].unique()))
    context_ids = list(range(len(unique_contexts)))
    contexts_mapping = dict(zip(unique_contexts, context_ids))
    ds["context_id"] = ds.apply(lambda x: contexts_mapping[x[context_column]], axis=1)

    stratify = None
    if stratify_column:
        assert stratify_column in ds.columns
        stratify = []
        for context_id in context_ids:
            df = ds[ds.context_id == context_id]
            value = df[stratify_column].values
            assert len(value.unique()) == 1
            stratify.append(value.unique()[0])

    train_indices, validation_indices = train_test_split(
        context_ids,
        train_size=train_size,
        stratify=stratify,
        random_state=seed,
    )

    dataset["train"] = Dataset.from_pandas(
        ds[ds.context_id.isin(train_indices)], preserve_index=False
    )

    if dev_size and test_size:
        dev_indices, test_indices = train_test_split(
            validation_indices,
            train_size=round(dev_size / (1 - train_size), 2),
            random_state=seed,
        )

        dataset["validation"] = Dataset.from_pandas(
            ds[ds.context_id.isin(dev_indices)], preserve_index=False
        )
        dataset["test"] = Dataset.from_pandas(
            ds[ds.context_id.isin(test_indices)], preserve_index=False
        )

    elif dev_size and not test_size:
        dataset["validation"] = Dataset.from_pandas(
            ds[ds.context_id.isin(validation_indices)], preserve_index=False
        )

    elif test_size and not dev_size:
        dataset["test"] = Dataset.from_pandas(
            ds[ds.context_id.isin(validation_indices)], preserve_index=False
        )

    return dataset