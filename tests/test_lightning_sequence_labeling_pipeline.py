from pathlib import Path
from typing import Any, Dict, Tuple

import datasets
import numpy as np
import pytest
import pytorch_lightning as pl
from _pytest.tmpdir import TempdirFactory

from embeddings.config.lightning_config import LightningAdvancedConfig
from embeddings.pipeline.hf_preprocessing_pipeline import HuggingFacePreprocessingPipeline
from embeddings.pipeline.lightning_pipeline import LightningPipeline
from embeddings.pipeline.lightning_sequence_labeling import LightningSequenceLabelingPipeline


@pytest.fixture(scope="module")
def tmp_path_module(tmpdir_factory: TempdirFactory) -> Path:
    path = tmpdir_factory.mktemp(__name__)
    return Path(path)


@pytest.fixture(scope="module")
def dataset_kwargs(tmp_path_module) -> Dict[str, Any]:
    path = str(tmp_path_module)
    pipeline = HuggingFacePreprocessingPipeline(
        dataset_name="clarin-pl/kpwr-ner",
        load_dataset_kwargs=None,
        persist_path=path,
        sample_missing_splits=None,
        ignore_test_subset=False,
        downsample_splits=(0.01, 0.01, 0.05),
        seed=441,
    )
    pipeline.run()

    return {
        "dataset_name_or_path": path,
        "input_column_name": "tokens",
        "target_column_name": "ner",
    }


@pytest.fixture(scope="module")
def config() -> LightningAdvancedConfig:
    return LightningAdvancedConfig(
        finetune_last_n_layers=0,
        task_train_kwargs={
            "max_epochs": 1,
            "devices": "auto",
            "accelerator": "cpu",
            "deterministic": True,
        },
        task_model_kwargs={
            "learning_rate": 1e-4,
            "train_batch_size": 32,
            "eval_batch_size": 32,
            "use_scheduler": False,
            "optimizer": "AdamW",
            "adam_epsilon": 1e-8,
            "warmup_steps": 100,
            "weight_decay": 0.0,
        },
        datamodule_kwargs={
            "max_seq_length": 64,
        },
        early_stopping_kwargs={
            "monitor": "val/Loss",
            "mode": "min",
            "patience": 3,
        },
        tokenizer_kwargs={},
        batch_encoding_kwargs={},
        dataloader_kwargs={},
        model_config_kwargs={},
    )


@pytest.fixture
def lightning_sequence_labeling_pipeline(
    dataset_kwargs: Dict[str, Any],
    config: LightningAdvancedConfig,
    tmp_path: Path,
) -> Tuple[LightningPipeline[datasets.DatasetDict, Dict[str, np.ndarray], Dict[str, Any]], Path]:
    return (
        LightningSequenceLabelingPipeline(
            output_path=tmp_path,
            embedding_name_or_path="allegro/herbert-base-cased",
            config=config,
            **dataset_kwargs,
        ),
        tmp_path,
    )


def test_lightning_sequence_labeling_pipeline(
    lightning_sequence_labeling_pipeline: Tuple[
        LightningPipeline[datasets.DatasetDict, Dict[str, np.ndarray], Dict[str, Any]],
        Path,
    ],
) -> None:
    pl.seed_everything(441)
    pipeline, path = lightning_sequence_labeling_pipeline
    result = pipeline.run()
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
    assert "y_probabilities" in result["data"]
    assert "names" in result["data"]
    assert isinstance(result["data"]["y_pred"], np.ndarray)
    assert isinstance(result["data"]["y_true"], np.ndarray)
    assert isinstance(result["data"]["y_probabilities"], np.ndarray)
    assert isinstance(result["data"]["names"], np.ndarray)
    assert isinstance(result["data"]["y_pred"][0], list)
    assert isinstance(result["data"]["y_true"][0], list)
    assert isinstance(result["data"]["y_probabilities"][0], list)
    assert isinstance(result["data"]["names"], np.ndarray)
    assert isinstance(result["data"]["y_pred"][0][0], str)
    assert isinstance(result["data"]["y_true"][0][0], str)
    assert isinstance(result["data"]["y_probabilities"][0][0], np.ndarray)
    assert isinstance(result["data"]["names"][0], str)
    assert isinstance(result["data"]["y_probabilities"][0][0][0], np.float32)
