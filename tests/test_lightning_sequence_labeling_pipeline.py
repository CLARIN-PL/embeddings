from pathlib import Path
from typing import Any, Dict, Tuple

import datasets
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from _pytest.tmpdir import TempdirFactory

from embeddings.config.lightning_config import LightningAdvancedConfig
from embeddings.evaluator.evaluation_results import Predictions, SequenceLabelingEvaluationResults
from embeddings.pipeline.hf_preprocessing_pipeline import HuggingFacePreprocessingPipeline
from embeddings.pipeline.lightning_pipeline import LightningPipeline
from embeddings.pipeline.lightning_sequence_labeling import LightningSequenceLabelingPipeline
from embeddings.task.lightning_task.sequence_labeling import SequenceLabelingTask


@pytest.fixture(scope="module")
def tmp_path_module(tmpdir_factory: TempdirFactory) -> Path:
    path = tmpdir_factory.mktemp(__name__)
    return Path(path)


@pytest.fixture(scope="module")
def dataset_kwargs(tmp_path_module: Path) -> Dict[str, Any]:
    pipeline = HuggingFacePreprocessingPipeline(
        dataset_name="clarin-pl/kpwr-ner",
        load_dataset_kwargs=None,
        persist_path=str(tmp_path_module),
        sample_missing_splits=None,
        ignore_test_subset=False,
        downsample_splits=(0.01, 0.01, 0.05),
        seed=441,
    )
    pipeline.run()

    return {
        "dataset_name_or_path": tmp_path_module,
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
    tmp_path_module: Path,
) -> LightningSequenceLabelingPipeline:
    return LightningSequenceLabelingPipeline(
        output_path=tmp_path_module,
        embedding_name_or_path="allegro/herbert-base-cased",
        config=config,
        **dataset_kwargs,
    )


def test_lightning_sequence_labeling_pipeline(
    lightning_sequence_labeling_pipeline: LightningSequenceLabelingPipeline,
    tmp_path_module: Path,
) -> None:
    pl.seed_everything(441)
    pipeline = lightning_sequence_labeling_pipeline
    result = pipeline.run()

    assert_result_values(result)
    assert_result_types(result)
    assert_inference_from_checkpoint(result, pipeline, tmp_path_module)


def assert_result_values(result: SequenceLabelingEvaluationResults) -> None:
    np.testing.assert_almost_equal(
        result.accuracy,
        0.0015690,
        decimal=pytest.decimal,
    )
    np.testing.assert_almost_equal(result.f1_micro, 0.0019846, decimal=pytest.decimal)
    np.testing.assert_almost_equal(
        result.precision_micro,
        0.0010559,
        decimal=pytest.decimal,
    )
    np.testing.assert_almost_equal(
        result.recall_micro,
        0.0164609,
        decimal=pytest.decimal,
    )


def assert_result_types(result: SequenceLabelingEvaluationResults) -> None:
    assert isinstance(result.data.y_true, np.ndarray)
    assert isinstance(result.data.y_probabilities, np.ndarray)
    assert isinstance(result.data.y_pred, np.ndarray)
    assert isinstance(result.data.names, np.ndarray)
    assert isinstance(result.data.y_pred[0], list)
    assert isinstance(result.data.y_true[0], list)
    assert isinstance(result.data.y_probabilities[0], list)
    assert isinstance(result.data.names, np.ndarray)
    assert isinstance(result.data.y_pred[0][0], str)
    assert isinstance(result.data.y_true[0][0], str)
    assert isinstance(result.data.y_probabilities[0][0], np.ndarray)
    assert isinstance(result.data.names[0], str)
    assert isinstance(result.data.y_probabilities[0][0][0], np.float32)


def assert_inference_from_checkpoint(
    result: Dict[str, Any],
    pipeline: LightningSequenceLabelingPipeline,
    tmp_path_module: Path,
) -> None:
    ckpt_path = tmp_path_module / "checkpoints" / "last.ckpt"
    task_from_ckpt = SequenceLabelingTask.from_checkpoint(
        checkpoint_path=ckpt_path.resolve(),
        output_path=tmp_path_module,
    )

    model_state_dict = pipeline.model.task.model.model.state_dict()
    model_from_ckpt_state_dict = task_from_ckpt.model.model.state_dict()
    assert model_state_dict.keys() == model_from_ckpt_state_dict.keys()
    for k in model_state_dict.keys():
        assert torch.equal(model_state_dict[k], model_from_ckpt_state_dict[k])

    predictions = task_from_ckpt.predict(pipeline.datamodule.test_dataloader())
    np.testing.assert_almost_equal(
        result.data.y_probabilities[0][0], predictions.y_probabilities[0][0]
    )
