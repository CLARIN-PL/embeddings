from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import datasets
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from _pytest.tmpdir import TempdirFactory

from embeddings.config.lightning_config import LightningAdvancedConfig
from embeddings.pipeline.hf_preprocessing_pipeline import HuggingFacePreprocessingPipeline
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline
from embeddings.pipeline.lightning_pipeline import LightningPipeline
from embeddings.task.lightning_task.text_classification import TextClassificationTask


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
        persist_path=tmp_path_module.name,
        sample_missing_splits=None,
        ignore_test_subset=False,
        downsample_splits=(0.01, 0.01, 0.05),
        seed=441,
    )
    pipeline.run()

    return {
        "dataset_name_or_path": tmp_path_module.name,
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
) -> LightningPipeline[datasets.DatasetDict, Dict[str, np.ndarray], Dict[str, Any]]:
    return LightningClassificationPipeline(
        embedding_name_or_path="allegro/herbert-base-cased",
        output_path=tmp_path_module.name,
        config=config,
        devices="auto",
        accelerator="cpu",
        **dataset_kwargs,
    )


def test_lightning_pipeline_inference(
    lightning_classification_pipeline: LightningPipeline[
        datasets.DatasetDict, Dict[str, np.ndarray], Dict[str, Any]
    ],
    tmp_path_module: "TemporaryDirectory[str]",
) -> None:
    pl.seed_everything(441, workers=True)
    pipeline = lightning_classification_pipeline
    results = pipeline.run()

    ckpt_path = Path(tmp_path_module.name) / "checkpoints" / "epoch=0-step=1.ckpt"
    task_from_ckpt = TextClassificationTask.from_checkpoint(
        checkpoint_path=ckpt_path.resolve(),
        output_path=tmp_path_module.name,
    )

    model_state_dict = pipeline.model.task.model.model.state_dict()
    model_from_ckpt_state_dict = task_from_ckpt.model.model.state_dict()
    assert model_state_dict.keys() == model_from_ckpt_state_dict.keys()
    for k in model_state_dict.keys():
        assert torch.equal(model_state_dict[k], model_from_ckpt_state_dict[k])

    test_dataloader = pipeline.datamodule.test_dataloader()
    predictions = task_from_ckpt.predict(test_dataloader, return_names=False)
    assert np.array_equal(results["data"]["y_probabilities"], predictions["y_probabilities"])
