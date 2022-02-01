from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import datasets
import numpy as np
import pytest
import pytorch_lightning as pl

from embeddings.pipeline.lightning_pipeline import LightningPipeline
from embeddings.pipeline.lightning_sequence_labeling import LightningSequenceLabelingPipeline


@pytest.fixture(scope="module")
def dataset_kwargs() -> Dict[str, Any]:
    return {
        "dataset_name_or_path": "clarin-pl/kpwr-ner",
        "input_column_name": "tokens",
        "target_column_name": "ner",
    }


@pytest.fixture(scope="module")
def pipeline_kwargs() -> Dict[str, Any]:
    return {"model_name_or_path": "allegro/herbert-base-cased", "finetune_last_n_layers": 0}


@pytest.fixture(scope="module")
def task_train_kwargs() -> Dict[str, Any]:
    return {
        "max_epochs": 1,
        "devices": "auto",
        "accelerator": "cpu",
        "deterministic": True,
    }


@pytest.fixture(scope="module")
def task_model_kwargs() -> Dict[str, Any]:
    return {"learning_rate": 5e-4, "use_scheduler": False}


@pytest.fixture(scope="module")
def datamodule_kwargs() -> Dict[str, Any]:
    return {
        "downsample_train": 0.01,
        "downsample_val": 0.01,
        "downsample_test": 0.05,
        "num_workers": 0,
    }


@pytest.fixture
def lightning_sequence_labeling_pipeline(
    pipeline_kwargs: Dict[str, Any],
    dataset_kwargs: Dict[str, Any],
    datamodule_kwargs: Dict[str, Any],
    task_train_kwargs: Dict[str, Any],
    result_path: "TemporaryDirectory[str]",
) -> Tuple[
    LightningPipeline[datasets.DatasetDict, Dict[str, np.ndarray], Dict[str, Any]],
    "TemporaryDirectory[str]",
]:
    datamodule_kwargs["max_seq_length"] = 64
    return (
        LightningSequenceLabelingPipeline(
            output_path=result_path.name,
            datamodule_kwargs=datamodule_kwargs,
            task_train_kwargs=task_train_kwargs,
            **pipeline_kwargs,
            **dataset_kwargs,
        ),
        result_path,
    )


def test_lightning_sequence_labeling_pipeline(
    lightning_sequence_labeling_pipeline: Tuple[
        LightningPipeline[datasets.DatasetDict, Dict[str, np.ndarray], Dict[str, Any]],
        "TemporaryDirectory[str]",
    ],
) -> None:
    pl.seed_everything(441)
    pipeline, path = lightning_sequence_labeling_pipeline
    result = pipeline.run()
    path.cleanup()
    np.testing.assert_almost_equal(
        result["seqeval__mode_None__scheme_None"]["overall_accuracy"],
        0.0015690,
        decimal=pytest.decimal,
    )
    np.testing.assert_almost_equal(
        result["seqeval__mode_None__scheme_None"]["overall_f1"], 0.0019846, decimal=pytest.decimal
    )
    np.testing.assert_almost_equal(
        result["seqeval__mode_None__scheme_None"]["overall_precision"],
        0.0010559,
        decimal=pytest.decimal,
    )
    np.testing.assert_almost_equal(
        result["seqeval__mode_None__scheme_None"]["overall_recall"],
        0.0164609,
        decimal=pytest.decimal,
    )

    assert "data" in result
    assert "y_pred" in result["data"]
    assert "y_true" in result["data"]
    assert isinstance(result["data"]["y_pred"], np.ndarray)
    assert isinstance(result["data"]["y_true"], np.ndarray)
    assert isinstance(result["data"]["y_pred"][0], list)
    assert isinstance(result["data"]["y_true"][0], list)
    assert isinstance(result["data"]["y_pred"][0][0], str)
    assert isinstance(result["data"]["y_true"][0][0], str)
