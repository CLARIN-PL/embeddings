import collections
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from unittest.mock import patch

import pandas as pd
import pytest
import yaml
from _pytest.tmpdir import TempdirFactory

from embeddings.hyperparameter_search.lighting_configspace import (
    LightingSequenceLabelingConfigSpace,
)
from embeddings.hyperparameter_search.parameters import ConstantParameter
from embeddings.pipeline.lightning_hps_pipeline import OptimizedLightingSequenceLabelingPipeline
from embeddings.pipeline.lightning_sequence_labeling import LightningSequenceLabelingPipeline
from embeddings.pipeline.pipelines_metadata import (
    LightningPipelineMetadata,
    LightningSequenceLabelingPipelineMetadata,
)

TESTING_DATAMODULE_KWARGS: Dict[str, Any] = deepcopy(
    LightningSequenceLabelingPipeline.DEFAULT_DATAMODULE_KWARGS
)
TESTING_DATAMODULE_KWARGS.update(
    {
        "downsample_train": 0.01,
        "downsample_val": 0.01,
        "downsample_test": 0.05,
    }
)


def _flatten(
    d: Union[collections.MutableMapping[Any, Any], LightningPipelineMetadata]
) -> Dict[Any, Any]:
    items: List[tuple[Any, Any]] = []
    for k, v in d.items():
        new_key = k
        if isinstance(v, collections.MutableMapping):
            items.extend(_flatten(v).items())
        else:
            items.append((new_key, v))
    return dict(items)


@pytest.fixture(scope="module")
def tmp_path_module(tmpdir_factory: TempdirFactory) -> Path:
    path = tmpdir_factory.mktemp(__name__)
    return Path(path)


@pytest.fixture(scope="module")
def retrain_tmp_path(tmp_path_module: Path) -> Path:
    path = tmp_path_module.joinpath("retrain")
    path.mkdir()
    return path


@pytest.fixture(scope="module")
def sequence_labelling_config_space() -> LightingSequenceLabelingConfigSpace:
    config_space = LightingSequenceLabelingConfigSpace(
        embedding_name="hf-internal-testing/tiny-albert",
        finetune_last_n_layers=ConstantParameter("finetune_last_n_layers", 0),
        max_epochs=ConstantParameter("max_epochs", 1),
    )
    return config_space


@pytest.fixture(scope="module")
@patch.object(
    LightningSequenceLabelingPipeline, "DEFAULT_DATAMODULE_KWARGS", TESTING_DATAMODULE_KWARGS
)
def sequence_labelling_hps_run_result(
    tmp_path_module: Path, sequence_labelling_config_space: LightingSequenceLabelingConfigSpace
) -> Tuple[pd.DataFrame, LightningSequenceLabelingPipelineMetadata]:
    assert LightningSequenceLabelingPipeline.DEFAULT_DATAMODULE_KWARGS == TESTING_DATAMODULE_KWARGS
    pipeline = OptimizedLightingSequenceLabelingPipeline(
        config_space=sequence_labelling_config_space,
        dataset_name="clarin-pl/kpwr-ner",
        input_column_name="tokens",
        target_column_name="ner",
        n_trials=1,
    ).persisting(
        best_params_path=tmp_path_module.joinpath("best_params.yaml"),
        log_path=tmp_path_module.joinpath("hps_log.pickle"),
    )
    df, metadata = pipeline.run()
    return df, metadata


def test_hps_output_files_exist(
    tmp_path_module: Path,
    sequence_labelling_hps_run_result: Tuple[
        pd.DataFrame, LightningSequenceLabelingPipelineMetadata
    ],
) -> None:
    assert tmp_path_module.joinpath("best_params.yaml").exists()
    assert tmp_path_module.joinpath("hps_log.pickle").exists()


def test_common_keys(
    sequence_labelling_hps_run_result: Tuple[
        pd.DataFrame, LightningSequenceLabelingPipelineMetadata
    ],
    sequence_labelling_config_space: LightingSequenceLabelingConfigSpace,
) -> None:
    df, metadata = sequence_labelling_hps_run_result
    assert _flatten(metadata).keys() & sequence_labelling_config_space._get_fields().keys() == {
        "learning_rate",
        "adam_epsilon",
        "max_epochs",
        "max_seq_length",
        "weight_decay",
        "finetune_last_n_layers",
        "warmup_steps",
        "use_scheduler",
        "optimizer",
        # "classifier_dropout",  # todo: uncomment
        "label_all_tokens",
    }


def test_keys_allowed_in_metadata_but_not_in_config_space(
    sequence_labelling_hps_run_result: Tuple[
        pd.DataFrame, LightningSequenceLabelingPipelineMetadata
    ],
    sequence_labelling_config_space: LightingSequenceLabelingConfigSpace,
) -> None:
    df, metadata = sequence_labelling_hps_run_result
    assert _flatten(metadata).keys() - sequence_labelling_config_space._get_fields().keys() == {
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
        "embedding_name",
        "tokenizer_name",
        "evaluation_mode",
        "tagging_scheme",
    }


def test_keys_allowed_in_config_space_but_not_in_metadata(
    sequence_labelling_hps_run_result: Tuple[
        pd.DataFrame, LightningSequenceLabelingPipelineMetadata
    ],
    sequence_labelling_config_space: LightingSequenceLabelingConfigSpace,
) -> None:

    df, metadata = sequence_labelling_hps_run_result
    assert sequence_labelling_config_space._get_fields().keys() - _flatten(metadata).keys() == {
        "trainer_devices",
        "param_embedding_name",
        "trainer_accelerator",
        "classifier_dropout",  # todo: remove 'classifier_dropout'
        "mini_batch_size",
    }


@pytest.fixture(scope="module")
@patch.object(
    LightningSequenceLabelingPipeline, "DEFAULT_DATAMODULE_KWARGS", TESTING_DATAMODULE_KWARGS
)
def retrain_model_result(
    retrain_tmp_path: Path,
    sequence_labelling_hps_run_result: Tuple[
        pd.DataFrame, LightningSequenceLabelingPipelineMetadata
    ],
) -> Dict[str, Any]:
    assert LightningSequenceLabelingPipeline.DEFAULT_DATAMODULE_KWARGS == TESTING_DATAMODULE_KWARGS
    df, metadata = sequence_labelling_hps_run_result
    metadata["output_path"] = retrain_tmp_path
    pipeline = LightningSequenceLabelingPipeline(**metadata)
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
    sequence_labelling_hps_run_result: Tuple[
        pd.DataFrame, LightningSequenceLabelingPipelineMetadata
    ],
) -> None:
    df, metadata = sequence_labelling_hps_run_result

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
        "tokenizer_name",
        "tokenizer_kwargs",
        "max_seq_length",
        "embedding_name",
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
        "evaluation_mode",
        "label_all_tokens",
        "tagging_scheme",
    }
    for k in common_keys:
        if k == "tokenizer_name":  # todo: remove
            continue
        if k in metadata:
            if isinstance(metadata[k], Enum):  # type: ignore
                enum_hparam = getattr(type(metadata[k]), hparams[k])  # type: ignore
                assert enum_hparam == best_params[k] == metadata[k]  # type: ignore
            else:
                assert hparams[k] == best_params[k] == metadata[k]  # type: ignore
        else:
            assert hparams[k] == best_params[k]

        if not k.endswith("_kwargs"):
            if k in ["max_seq_length", "tagging_scheme"]:
                continue
            assert hparams[k] is not None

    assert hparams["model_name_or_path"] == best_params["embedding_name"]
