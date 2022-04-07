from pathlib import Path
from typing import Any, Dict, Tuple

import datasets
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from _pytest.tmpdir import TempdirFactory

from embeddings.pipeline.hf_preprocessing_pipeline import HuggingFacePreprocessingPipeline
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline
from embeddings.pipeline.lightning_pipeline import LightningPipeline
from embeddings.task.lightning_task.text_classification import TextClassificationTask


@pytest.fixture(scope="module")
def tmp_path_module(tmpdir_factory: TempdirFactory) -> Path:
    path = tmpdir_factory.mktemp(__name__)
    return Path(path)


@pytest.fixture(scope="module")
def pipeline_kwargs() -> Dict[str, Any]:
    return {
        "embedding_name_or_path": "hf-internal-testing/tiny-albert",
        "finetune_last_n_layers": 0,
    }


@pytest.fixture(scope="module")
def dataset_kwargs(tmp_path_module) -> Dict[str, Any]:
    path = str(tmp_path_module)
    pipeline = HuggingFacePreprocessingPipeline(
        dataset_name="clarin-pl/polemo2-official",
        load_dataset_kwargs={
            "train_domains": ["hotels", "medicine"],
            "dev_domains": ["hotels", "medicine"],
            "test_domains": ["hotels", "medicine"],
            "text_cfg": "text",
        },
        persist_path=path,
        sample_missing_splits=None,
        ignore_test_subset=False,
        downsample_splits=(0.01, 0.01, 0.05),
        seed=441,
    )
    pipeline.run()

    return {
        "dataset_name_or_path": path,
        "input_column_name": ["text"],
        "target_column_name": "target",
    }


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
    return {"num_workers": 0}


@pytest.fixture(scope="module")
def lightning_classification_pipeline(
    pipeline_kwargs: Dict[str, Any],
    dataset_kwargs: Dict[str, Any],
    datamodule_kwargs: Dict[str, Any],
    task_train_kwargs: Dict[str, Any],
    task_model_kwargs: Dict[str, Any],
    result_path: Path,
) -> Tuple[LightningPipeline[datasets.DatasetDict, Dict[str, np.ndarray], Dict[str, Any]], Path]:
    return (
        LightningClassificationPipeline(
            output_path=result_path.name,
            **pipeline_kwargs,
            **dataset_kwargs,
            datamodule_kwargs=datamodule_kwargs,
            task_train_kwargs=task_train_kwargs,
            task_model_kwargs=task_model_kwargs,
        ),
        result_path,
    )


def test_lightning_pipeline_inference(
    lightning_classification_pipeline: Tuple[
        LightningPipeline[datasets.DatasetDict, Dict[str, np.ndarray], Dict[str, Any]],
        Path,
    ],
) -> None:
    pl.seed_everything(441, workers=True)

    pipeline, path = lightning_classification_pipeline
    results = pipeline.run()

    ckpt_path = Path(path.name) / "checkpoints" / "epoch=0-step=1.ckpt"
    task_from_ckpt = TextClassificationTask.from_checkpoint(
        checkpoint_path=ckpt_path.resolve(),
        output_path=path.name,
    )

    model_state_dict = pipeline.model.task.model.model.state_dict()
    model_from_ckpt_state_dict = task_from_ckpt.model.model.state_dict()

    assert model_state_dict.keys() == model_from_ckpt_state_dict.keys()
    for k in model_state_dict.keys():
        assert torch.equal(model_state_dict[k], model_from_ckpt_state_dict[k])

    test_dataloader = pipeline.datamodule.test_dataloader()
    predictions = task_from_ckpt.predict(test_dataloader, return_names=False)

    assert np.array_equal(results["data"]["y_probabilities"], predictions["y_probabilities"])
