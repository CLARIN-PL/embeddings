from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import datasets
import numpy as np
import pandas as pd
import pytest
from numpy import typing as nptyping
from sklearn.ensemble import AdaBoostClassifier

from embeddings.pipeline.sklearn_classification import SklearnClassificationPipeline


@pytest.fixture(scope="module")
def dataset_kwargs() -> Dict[str, Any]:
    return {
        "dataset_name": "clarin-pl/polemo2-official",
        "input_column_name": "text",
        "target_column_name": "target",
    }


@pytest.fixture(scope="module")
def embedding_kwargs() -> Dict[str, Any]:
    return {"max_features": 100}


@pytest.fixture(scope="module")
def sklearn_classification_pipeline(
    dataset_kwargs: Dict[str, any],
    embedding_kwargs: Dict[str, Any],
    result_path: "TemporaryDirectory[str]",
) -> Tuple[SklearnClassificationPipeline, "TemporaryDirectory[str]"]:
    return (
        SklearnClassificationPipeline(
            **dataset_kwargs,
            embedding_kwargs=embedding_kwargs,
            output_path=Path(result_path.name),
            classifier=AdaBoostClassifier
        ),
        result_path,
    )


def test_sklearn_classification_pipeline(
    sklearn_classification_pipeline: Tuple[
        SklearnClassificationPipeline,
        "TemporaryDirectory[str]",
    ],
) -> None:
    pipeline, path = sklearn_classification_pipeline
    result = pipeline.run()
    path.cleanup()
    np.testing.assert_almost_equal(result["accuracy"]["accuracy"], 0.62317, decimal=pytest.decimal)
    np.testing.assert_almost_equal(
        result["f1__average_macro"]["f1"], 0.60027, decimal=pytest.decimal
    )
    np.testing.assert_almost_equal(
        result["precision__average_macro"]["precision"], 0.59270, decimal=pytest.decimal
    )
    np.testing.assert_almost_equal(
        result["recall__average_macro"]["recall"], 0.61276, decimal=pytest.decimal
    )

    assert "data" in result
    assert "y_pred" in result["data"]
    assert "y_true" in result["data"]
    assert isinstance(result["data"]["y_pred"], np.ndarray)
    assert isinstance(result["data"]["y_true"], np.ndarray)
    assert result["data"]["y_pred"].dtype == np.int64
    assert result["data"]["y_true"].dtype == np.int64
