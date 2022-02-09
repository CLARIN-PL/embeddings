from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Tuple
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from embeddings.hyperparameter_search.lighting_configspace import (
    LightingTextClassificationConfigSpace,
)
from embeddings.hyperparameter_search.parameters import ConstantParameter
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline
from embeddings.pipeline.lightning_hps_pipeline import OptimizedLightingClassificationPipeline
from embeddings.pipeline.pipelines_metadata import LightningClassificationPipelineMetadata
from tests.hps.utils import _flatten

TESTING_DATAMODULE_KWARGS: Dict[str, Any] = deepcopy(
    LightningClassificationPipeline.DEFAULT_DATAMODULE_KWARGS
)
TESTING_DATAMODULE_KWARGS.update(
    {
        "downsample_train": 0.01,
        "downsample_val": 0.01,
        "downsample_test": 0.05,
    }
)


@pytest.fixture(scope="module")
def classification_config_space() -> LightingTextClassificationConfigSpace:
    config_space = LightingTextClassificationConfigSpace(
        embedding_name_or_path="hf-internal-testing/tiny-albert",
        finetune_last_n_layers=ConstantParameter("finetune_last_n_layers", 0),
        max_epochs=ConstantParameter("max_epochs", 1),
    )
    return config_space


@pytest.fixture(scope="module")
@patch.object(
    LightningClassificationPipeline, "DEFAULT_DATAMODULE_KWARGS", TESTING_DATAMODULE_KWARGS
)
def classification_hps_run_result(
    tmp_path_module: Path, classification_config_space: LightingTextClassificationConfigSpace
) -> Tuple[pd.DataFrame, LightningClassificationPipelineMetadata]:
    assert LightningClassificationPipeline.DEFAULT_DATAMODULE_KWARGS == TESTING_DATAMODULE_KWARGS
    pipeline = OptimizedLightingClassificationPipeline(
        config_space=classification_config_space,
        dataset_name="clarin-pl/polemo2-official",
        input_column_name="text",
        target_column_name="target",
        n_trials=1,
    ).persisting(
        best_params_path=tmp_path_module.joinpath("best_params.yaml"),
        log_path=tmp_path_module.joinpath("hps_log.pickle"),
    )
    df, metadata = pipeline.run()
    return df, metadata


def test_hps_output_files_exist(
    tmp_path_module: Path,
    classification_hps_run_result: Tuple[pd.DataFrame, LightningClassificationPipelineMetadata],
) -> None:
    assert tmp_path_module.joinpath("best_params.yaml").exists()
    assert tmp_path_module.joinpath("hps_log.pickle").exists()


def test_common_keys(
    classification_hps_run_result: Tuple[pd.DataFrame, LightningClassificationPipelineMetadata],
    classification_config_space: LightingTextClassificationConfigSpace,
) -> None:
    df, metadata = classification_hps_run_result
    assert _flatten(metadata).keys() & classification_config_space._get_fields().keys() == {
        "learning_rate",
        "adam_epsilon",
        "max_epochs",
        "max_seq_length",
        "weight_decay",
        "finetune_last_n_layers",
        "warmup_steps",
        "use_scheduler",
        "optimizer",
        "classifier_dropout"
    }


def test_keys_allowed_in_metadata_but_not_in_config_space(
    classification_hps_run_result: Tuple[pd.DataFrame, LightningClassificationPipelineMetadata],
    classification_config_space: LightingTextClassificationConfigSpace,
) -> None:
    df, metadata = classification_hps_run_result
    assert _flatten(metadata).keys() - classification_config_space.__dict__.keys() == {
        "tokenizer_kwargs",
        "batch_encoding_kwargs",
        "train_batch_size",
        "predict_subset",
        "input_column_name",
        "load_dataset_kwargs",
        "target_column_name",
        "accelerator",
        "dataset_name_or_path",
        "eval_batch_size",
        "devices",
        "embedding_name_or_path",
        "tokenizer_name_or_path",
    }


def test_keys_allowed_in_config_space_but_not_in_metadata(
    classification_hps_run_result: Tuple[pd.DataFrame, LightningClassificationPipelineMetadata],
    classification_config_space: LightingTextClassificationConfigSpace,
) -> None:

    df, metadata = classification_hps_run_result
    assert classification_config_space.__dict__.keys() - _flatten(metadata).keys() == {
        "trainer_devices",
        "param_embedding_name_or_path",
        "trainer_accelerator",
        "mini_batch_size",
    }


@pytest.fixture(scope="module")
@patch.object(
    LightningClassificationPipeline, "DEFAULT_DATAMODULE_KWARGS", TESTING_DATAMODULE_KWARGS
)
def retrain_model_result(
    retrain_tmp_path: Path,
    classification_hps_run_result: Tuple[pd.DataFrame, LightningClassificationPipelineMetadata],
) -> Dict[str, Any]:
    assert LightningClassificationPipeline.DEFAULT_DATAMODULE_KWARGS == TESTING_DATAMODULE_KWARGS
    df, metadata = classification_hps_run_result
    metadata["output_path"] = retrain_tmp_path
    pipeline = LightningClassificationPipeline(**metadata)
    results = pipeline.run()
    return results


def test_evaluation_json_exists(
    retrain_tmp_path: Path, retrain_model_result: Dict[str, Any]
) -> None:
    assert retrain_tmp_path.joinpath("evaluation.json").exists()


def test_hparams_best_params_files_compatibility(
    tmp_path_module: Path,
    retrain_tmp_path: Path,
    retrain_model_result: Dict[str, Any],
    classification_hps_run_result: Tuple[pd.DataFrame, LightningClassificationPipelineMetadata],
) -> None:
    df, metadata = classification_hps_run_result

    with open(retrain_tmp_path / "lightning_logs" / "version_0" / "hparams.yaml") as f:
        hparams = yaml.load(f, Loader=yaml.Loader)

    with open(tmp_path_module / "best_params.yaml") as f:
        best_params = yaml.load(f, Loader=yaml.Loader)

    hparams = _flatten(hparams)
    best_params = _flatten(best_params)

    common_keys = hparams.keys() & best_params.keys()

    assert common_keys == {
        "train_batch_size",
        "batch_encoding_kwargs",
        "tokenizer_name_or_path",
        "tokenizer_kwargs",
        "max_seq_length",
        "embedding_name_or_path",
        "finetune_last_n_layers",
        "dataset_name_or_path",
        "input_column_name",
        "weight_decay",
        "use_scheduler",
        "devices",
        "eval_batch_size",
        "adam_epsilon",
        "optimizer",
        "learning_rate",
        "load_dataset_kwargs",
        "max_epochs",
        "predict_subset",
        "warmup_steps",
        "target_column_name",
        "accelerator",
    }
    for k in common_keys:
        if k in metadata:
            if isinstance(metadata[k], Enum):  # type: ignore
                enum_hparam = getattr(type(metadata[k]), hparams[k])  # type: ignore
                assert enum_hparam == best_params[k] == metadata[k]  # type: ignore
            else:
                assert hparams[k] == best_params[k] == metadata[k]  # type: ignore
        else:
            assert hparams[k] == best_params[k]

        if not k.endswith("_kwargs") and k != "max_seq_length":
            assert hparams[k] is not None

    assert hparams["model_name_or_path"] == best_params["embedding_name_or_path"]
