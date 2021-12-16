from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import datasets
import numpy as np
import pytest

from embeddings.pipeline.lightning_classification import LightningClassificationPipeline
from embeddings.pipeline.lightning_pipeline import LightningPipeline


@pytest.fixture
def pipeline_kwargs() -> Dict[str, Any]:
    return {
        "embedding_name": "allegro/herbert-base-cased",
    }


@pytest.fixture
def dataset_kwargs() -> Dict[str, Any]:
    return {
        "embedding_name": "allegro/herbert-base-cased",
        "dataset_name": "clarin-pl/polemo2-official",
        "input_column_name": ["text"],
        "target_column_name": "target",
        "load_dataset_kwargs": {
            "train_domains": ["hotels", "medicine"],
            "dev_domains": ["hotels", "medicine"],
            "test_domains": ["hotels", "medicine"],
            "text_cfg": "text",
        },
    }


@pytest.fixture
def task_train_kwargs() -> Dict[str, Any]:
    return {
        "max_epochs": 1,
        "devices": "auto",
        "accelerator": "auto",
        "limit_train_batches": 0.1,
        "limit_val_batches": 0.1,
        "limit_test_batches": 0.1,
    }


@pytest.fixture
def task_model_kwargs() -> Dict[str, Any]:
    return {"learning_rate": 5e-4}


@pytest.fixture
def lightning_classification_pipeline(
    pipeline_kwargs: Dict[str, Any],
    dataset_kwargs: Dict[str, Any],
    task_train_kwargs: Dict[str, Any],
    task_model_kwargs: Dict[str, Any],
    output_path: "TemporaryDirectory[str]",
) -> Tuple[
    LightningPipeline[datasets.DatasetDict, Dict[str, np.ndarray], Dict[str, Any]],
    "TemporaryDirectory[str]",
]:
    return (
        LightningClassificationPipeline(
            output_path=output_path.name,
            **pipeline_kwargs,
            **dataset_kwargs,
            task_train_kwargs=task_train_kwargs,
            task_model_kwargs=task_model_kwargs,
        ),
        output_path,
    )


def test_lightning_classification_pipeline(
    lightning_classification_pipeline: Tuple[
        LightningPipeline[datasets.DatasetDict, Dict[str, np.ndarray], Dict[str, Any]],
        "TemporaryDirectory[str]",
    ],
) -> None:
    pipeline, path = lightning_classification_pipeline
    pipeline.run()
    path.cleanup()
    # np.testing.assert_almost_equal(result["accuracy"]["accuracy"], 0.3333333)
    # np.testing.assert_almost_equal(result["f1__average_macro"]["f1"], 0.1666666)
    # np.testing.assert_almost_equal(result["precision__average_macro"]["precision"], 0.1111111)
    # np.testing.assert_almost_equal(result["recall__average_macro"]["recall"], 0.3333333)
