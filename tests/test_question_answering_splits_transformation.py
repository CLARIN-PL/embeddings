from typing import Dict, Any

import numpy as np
import pandas as pd

from embeddings.transformation.hf_transformation.qa_data_split_transformation import (
    QuestionAnsweringSplitsTransformation,
)

import datasets
import pytest


@pytest.fixture(scope="module")
def split_kwargs() -> Dict[str, Any]:
    return {"train_size": 0.7, "dev_size": 0.15, "test_size": 0.15, "seed": 42}


@pytest.fixture(scope="module")
def squad_v2_dataset() -> datasets.Dataset:
    dataset_dict = datasets.load_dataset("squad_v2")
    concatenated_df = pd.concat(
        [dataset_dict[key].to_pandas() for key in dataset_dict.keys()], axis=0
    )
    return datasets.Dataset.from_pandas(concatenated_df)


@pytest.fixture(scope="module")
def qa_data_split_transformation(
    split_kwargs: Dict[str, Any],
) -> QuestionAnsweringSplitsTransformation:
    return QuestionAnsweringSplitsTransformation(**split_kwargs)


def test_question_answering_splits_transformation(
    qa_data_split_transformation: QuestionAnsweringSplitsTransformation,
    squad_v2_dataset: datasets.Dataset,
):
    transformation = qa_data_split_transformation
    dataset = squad_v2_dataset
    split_result = transformation.transform(dataset)
    train_context_ids = set(split_result["train"]["context_id"])
    validation_context_ids = set(split_result["validation"]["context_id"])
    test_context_ids = set(split_result["test"]["context_id"])

    assert isinstance(split_result, datasets.DatasetDict)
    assert len(split_result.keys()) == 3
    np.testing.assert_almost_equal(
        split_result["train"].num_rows / dataset.num_rows, 0.7, decimal=pytest.decimal
    )
    np.testing.assert_almost_equal(
        split_result["validation"].num_rows / dataset.num_rows, 0.15, decimal=pytest.decimal
    )
    np.testing.assert_almost_equal(
        split_result["test"].num_rows / dataset.num_rows, 0.15, decimal=pytest.decimal
    )
    assert len(train_context_ids.intersection(validation_context_ids)) == 0
    assert len(train_context_ids.intersection(test_context_ids)) == 0
    assert len(validation_context_ids.intersection(test_context_ids)) == 0
