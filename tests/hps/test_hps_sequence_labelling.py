from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import pandas as pd
import pytest
import yaml
from _pytest.tmpdir import TempdirFactory

from embeddings.config.lighting_config_space import LightingSequenceLabelingConfigSpace
from embeddings.config.lightning_config import LightningConfigDefaultKwargs
from embeddings.config.parameters import ConstantParameter
from embeddings.pipeline.hf_preprocessing_pipeline import HuggingFacePreprocessingPipeline
from embeddings.pipeline.lightning_hps_pipeline import OptimizedLightingSequenceLabelingPipeline
from embeddings.pipeline.lightning_sequence_labeling import LightningSequenceLabelingPipeline
from embeddings.pipeline.pipelines_metadata import LightningSequenceLabelingPipelineMetadata
from embeddings.utils.loggers import LightningLoggingConfig
from tests.hps.utils import _flatten


@pytest.fixture(scope="module")
def dataset_name() -> str:
    return "clarin-pl/kpwr-ner"


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
    metadata_config_keys = _flatten(metadata["config"].__dict__)
    config_space_keys = sequence_labelling_config_space._get_fields().keys()
    assert metadata_config_keys & config_space_keys == {
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
    metadata_keys = {**metadata, **metadata["config"].__dict__}.keys()
    config_space_keys = sequence_labelling_config_space.__dict__.keys()
    assert metadata_keys - config_space_keys == {
        "task_train_kwargs",
        "embedding_name_or_path",
        "config",
        "predict_subset",
        "model_config_kwargs",
        "input_column_name",
        "early_stopping_kwargs",
        "load_dataset_kwargs",
        "tokenizer_kwargs",
        "dataset_name_or_path",
        "tokenizer_name_or_path",
        "batch_encoding_kwargs",
        "datamodule_kwargs",
        "eval_batch_size",
        "train_batch_size",
        "tagging_scheme",
        "task_model_kwargs",
        "evaluation_mode",
        "target_column_name",
        "dataloader_kwargs",
    }


def test_keys_allowed_in_config_space_but_not_in_metadata(
    sequence_labelling_hps_run_result: Tuple[
        pd.DataFrame, LightningSequenceLabelingPipelineMetadata
    ],
    sequence_labelling_config_space: LightingSequenceLabelingConfigSpace,
) -> None:
    df, metadata = sequence_labelling_hps_run_result
    metadata_keys = {**metadata, **_flatten(metadata["config"].__dict__)}.keys()
    config_space_keys = sequence_labelling_config_space.__dict__.keys()
    assert config_space_keys - metadata_keys == {
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

    # compare config keys
    assert (
        best_params["config"].__dict__.keys()
        == hparams["config"].__dict__.keys()
        == metadata["config"].__dict__.keys()
    )
    # compare config values
    assert (
        _flatten(best_params["config"].__dict__)
        == _flatten(hparams["config"].__dict__)
        == _flatten(metadata["config"].__dict__)
    )

    assert hparams["model_name_or_path"] == best_params["embedding_name_or_path"]
    hparams = {k: v for k, v in hparams.items() if k in best_params.keys() and k != "config"}
    for k in hparams.keys():
        if isinstance(metadata[k], Enum):  # type: ignore
            enum_hparam = getattr(type(metadata[k]), hparams[k])  # type: ignore
            assert enum_hparam == best_params[k] == metadata[k]  # type: ignore
        else:
            assert hparams[k] == best_params[k] == metadata[k]  # type: ignore
