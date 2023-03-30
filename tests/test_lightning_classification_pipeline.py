from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from _pytest.tmpdir import TempdirFactory

from embeddings.config.lightning_config import LightningAdvancedConfig
from embeddings.evaluator.evaluation_results import TextClassificationEvaluationResults
from embeddings.pipeline.hf_preprocessing_pipeline import HuggingFacePreprocessingPipeline
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline
from embeddings.task.lightning_task.text_classification import TextClassificationTaskClassification


@pytest.fixture(scope="module")
def tmp_path_module(tmpdir_factory: TempdirFactory) -> Path:
    path = tmpdir_factory.mktemp(__name__)
    return Path(path)


@pytest.fixture(scope="module")
def dataset_kwargs(tmp_path_module: Path) -> Dict[str, Any]:
    pipeline = HuggingFacePreprocessingPipeline(
        dataset_name="clarin-pl/polemo2-official",
        load_dataset_kwargs={
            "train_domains": ["hotels", "medicine"],
            "dev_domains": ["hotels", "medicine"],
            "test_domains": ["hotels", "medicine"],
            "text_cfg": "text",
        },
        persist_path=str(tmp_path_module),
        sample_missing_splits=None,
        ignore_test_subset=False,
        downsample_splits=(0.01, 0.01, 0.05),
        seed=441,
    )
    pipeline.run()

    return {
        "dataset_name_or_path": tmp_path_module,
        "input_column_name": ["text"],
        "target_column_name": "target",
    }


@pytest.fixture(scope="module")
def config() -> LightningAdvancedConfig:
    return LightningAdvancedConfig(
        finetune_last_n_layers=0,
        task_train_kwargs={
            "max_epochs": 1,
            "deterministic": True,
        },
        task_model_kwargs={
            "learning_rate": 5e-4,
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


@pytest.fixture(scope="module")
def lightning_classification_pipeline(
    dataset_kwargs: Dict[str, Any],
    config: LightningAdvancedConfig,
    tmp_path_module: Path,
) -> LightningClassificationPipeline:
    return LightningClassificationPipeline(
        embedding_name_or_path="allegro/herbert-base-cased",
        output_path=tmp_path_module,
        config=config,
        devices="auto",
        accelerator="cpu",
        model_checkpoint_kwargs={
            "filename": "last",
            "monitor": None,
            "save_last": False,
        },
        **dataset_kwargs,
    )


def test_lightning_classification_pipeline(
    lightning_classification_pipeline: LightningClassificationPipeline,
    tmp_path_module: Path,
) -> None:
    pl.seed_everything(441, workers=True)
    pipeline = lightning_classification_pipeline
    result = pipeline.run()

    assert_result_values(result)
    assert_result_types(result)
    assert_inference_from_checkpoint(result, pipeline, tmp_path_module)


def assert_result_values(result: TextClassificationEvaluationResults) -> None:
    np.testing.assert_almost_equal(result.accuracy, 0.3783783, decimal=pytest.decimal)
    np.testing.assert_almost_equal(result.f1_macro, 0.1399999, decimal=pytest.decimal)
    np.testing.assert_almost_equal(result.precision_macro, 0.1, decimal=pytest.decimal)
    np.testing.assert_almost_equal(result.recall_macro, 0.2333333, decimal=pytest.decimal)


def assert_result_types(result: TextClassificationEvaluationResults) -> None:
    assert isinstance(result, TextClassificationEvaluationResults)
    assert isinstance(result.data.y_pred, np.ndarray)
    assert isinstance(result.data.y_true, np.ndarray)
    assert isinstance(result.data.y_probabilities, np.ndarray)
    assert isinstance(result.data.names, np.ndarray)
    assert result.data.y_pred.dtype == np.int64
    assert result.data.y_true.dtype == np.int64
    assert result.data.y_probabilities.dtype == np.float32
    assert isinstance(result.data.names[0], str)


def assert_inference_from_checkpoint(
    result: TextClassificationEvaluationResults,
    pipeline: LightningClassificationPipeline,
    tmp_path_module: Path,
) -> None:
    ckpt_path = tmp_path_module / "checkpoints" / "last.ckpt"
    task_from_ckpt = TextClassificationTaskClassification.from_checkpoint(
        checkpoint_path=ckpt_path.resolve(),
        output_path=tmp_path_module,
    )

    model_state_dict = pipeline.model.task.model.model.state_dict()
    model_from_ckpt_state_dict = task_from_ckpt.model.model.state_dict()
    assert model_state_dict.keys() == model_from_ckpt_state_dict.keys()
    for k in model_state_dict.keys():
        assert torch.equal(model_state_dict[k], model_from_ckpt_state_dict[k])

    predictions = task_from_ckpt.predict(pipeline.datamodule.test_dataloader())
    assert np.array_equal(result.data.y_probabilities, predictions.y_probabilities)
