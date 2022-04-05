from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import pandas as pd
import pytest
import yaml
from _pytest.tmpdir import TempdirFactory

from embeddings.hyperparameter_search.lighting_configspace import (
    LightingSequenceLabelingConfigSpace,
)
from embeddings.hyperparameter_search.parameters import ConstantParameter
from embeddings.pipeline.hf_preprocessing_pipeline import HuggingFacePreprocessingPipeline
from embeddings.pipeline.lightning_hps_pipeline import OptimizedLightingSequenceLabelingPipeline
from embeddings.pipeline.lightning_sequence_labeling import LightningSequenceLabelingPipeline
from embeddings.pipeline.pipelines_metadata import LightningSequenceLabelingPipelineMetadata
from embeddings.utils.loggers import LightningLoggingConfig
from tests.hps.utils import _flatten


@pytest.fixture(scope="module")
def dataset_name() -> str:
    return "clarin-pl/kpwr-ner"


@pytest.fixture(scope="module")
def hps_dataset_path(dataset_name: str) -> "TemporaryDirectory[str]":
    path = TemporaryDirectory()
    pipeline = HuggingFacePreprocessingPipeline(
        dataset_name=dataset_name,
        load_dataset_kwargs=None,
        persist_path=path.name,
        sample_missing_splits=(0.1, None),
        ignore_test_subset=True,
        downsample_splits=(0.01, 0.01, 0.05),
        seed=441,
    )
    pipeline.run()

    return path


@pytest.fixture(scope="module")
def dataset_path(dataset_name: str) -> "TemporaryDirectory[str]":
    path = TemporaryDirectory()
    pipeline = HuggingFacePreprocessingPipeline(
        dataset_name=dataset_name,
        load_dataset_kwargs=None,
        persist_path=path.name,
        sample_missing_splits=None,
        ignore_test_subset=False,
        downsample_splits=(0.01, 0.01, 0.05),
        seed=441,
    )
    pipeline.run()

    return path


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
        embedding_name_or_path="hf-internal-testing/tiny-albert",
        finetune_last_n_layers=ConstantParameter("finetune_last_n_layers", 0),
        max_epochs=ConstantParameter("max_epochs", 1),
    )
    return config_space


@pytest.fixture(scope="module")
def sequence_labelling_hps_run_result(
    hps_dataset_path: "TemporaryDirectory[str]",
    tmp_path_module: Path,
    sequence_labelling_config_space: LightingSequenceLabelingConfigSpace,
) -> Tuple[pd.DataFrame, LightningSequenceLabelingPipelineMetadata]:
    pipeline = OptimizedLightingSequenceLabelingPipeline(
        config_space=sequence_labelling_config_space,
        dataset_name_or_path=hps_dataset_path.name,
        input_column_name="tokens",
        target_column_name="ner",
        n_trials=1,
        ignore_preprocessing_pipeline=True,
        logging_config=LightningLoggingConfig.from_flags(csv=True),
    ).persisting(
        best_params_path=tmp_path_module.joinpath("best_params.yaml"),
        log_path=tmp_path_module.joinpath("hps_log.pickle"),
    )
    df, metadata = pipeline.run(catch=())
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
        "label_all_tokens",
        "classifier_dropout",
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
        "embedding_name_or_path",
        "tokenizer_name_or_path",
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
        "param_embedding_name_or_path",
        "trainer_accelerator",
        "mini_batch_size",
    }


@pytest.fixture(scope="module")
def retrain_model_result(
    dataset_path: "TemporaryDirectory[str]",
    retrain_tmp_path: Path,
    sequence_labelling_hps_run_result: Tuple[
        pd.DataFrame, LightningSequenceLabelingPipelineMetadata
    ],
) -> Dict[str, Any]:
    df, metadata = sequence_labelling_hps_run_result
    metadata["dataset_name_or_path"] = dataset_path.name
    metadata["output_path"] = retrain_tmp_path
    pipeline = LightningSequenceLabelingPipeline(
        logging_config=LightningLoggingConfig.from_flags(csv=True), **metadata
    )
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

    with open(retrain_tmp_path / "csv" / "version_0" / "hparams.yaml") as f:
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
        "evaluation_mode",
        "label_all_tokens",
        "tagging_scheme",
        "classifier_dropout",
    }

    for k in common_keys:
        if k == "dataset_name_or_path":
            continue
        if k in metadata:
            if isinstance(metadata[k], Enum):  # type: ignore
                enum_hparam = getattr(type(metadata[k]), hparams[k])  # type: ignore
                assert enum_hparam == best_params[k] == metadata[k]  # type: ignore
            else:
                assert hparams[k] == best_params[k] == metadata[k]  # type: ignore
        else:
            assert hparams[k] == best_params[k]

        if not k.endswith("_kwargs") and k not in ["max_seq_length", "tagging_scheme"]:
            assert hparams[k] is not None

    assert hparams["model_name_or_path"] == best_params["embedding_name_or_path"]
