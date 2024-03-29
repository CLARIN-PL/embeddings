from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import pandas as pd
import pytest
import yaml
from _pytest.tmpdir import TempdirFactory

from embeddings.config.lighting_config_space import LightingSequenceLabelingConfigSpace
from embeddings.config.parameters import ConstantParameter
from embeddings.evaluator.evaluation_results import SequenceLabelingEvaluationResults
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
        "accelerator",
        "devices",
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
        "dataloader_kwargs",
        "tagging_scheme",
        "task_model_kwargs",
        "evaluation_mode",
        "target_column_name",
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
) -> SequenceLabelingEvaluationResults:
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

    with open(retrain_tmp_path / "pipeline_config.yaml") as f:
        hparams = yaml.load(f, Loader=yaml.Loader)

    with open(tmp_path_module / "best_params.yaml") as f:
        best_params = yaml.load(f, Loader=yaml.Loader)

    common_keys = _flatten(hparams).keys() & _flatten(best_params).keys()
    assert common_keys == {
        "config",
        "dataset_name_or_path",
        "tokenizer_name_or_path",
        "load_dataset_kwargs",
        "target_column_name",
        "predict_subset",
        "embedding_name_or_path",
        "input_column_name",
        "accelerator",
        "tagging_scheme",
        "devices",
        "evaluation_mode",
    }

    assert (
        best_params["config"].__dict__.keys()
        == hparams["config"].__dict__.keys()
        == metadata["config"].__dict__.keys()
    )

    assert_compare_config_values(best_params, hparams, metadata)
    assert_compare_params_values(best_params, hparams, metadata)


def assert_compare_config_values(
    best_params: Dict[str, Any],
    hparams: Dict[str, Any],
    metadata: LightningSequenceLabelingPipelineMetadata,
) -> None:
    best_params_config = _flatten(best_params["config"].__dict__)
    best_params_config.update(
        {"devices": best_params["devices"], "accelerator": best_params["accelerator"]}
    )  # append devices and accelerator to best params since this is done in lightning pipeline
    hparams_config = _flatten(hparams["config"].__dict__)
    metadata_config = _flatten(metadata["config"].__dict__)
    assert best_params_config == hparams_config == metadata_config


def assert_compare_params_values(
    best_params: Dict[str, Any],
    hparams: Dict[str, Any],
    metadata: LightningSequenceLabelingPipelineMetadata,
) -> None:
    assert hparams["embedding_name_or_path"] == best_params["embedding_name_or_path"]
    hparams = {k: v for k, v in hparams.items() if k in best_params.keys() and k != "config"}
    for k in hparams.keys():
        if k == "dataset_name_or_path":
            continue
        elif not k.endswith("_kwargs") and k not in ["tagging_scheme"]:
            assert hparams[k] is not None
        elif isinstance(metadata[k], Enum):
            enum_hparam = getattr(type(metadata[k]), hparams[k])
            assert enum_hparam == best_params[k] == metadata[k]
        else:
            assert hparams[k] == best_params[k] == metadata[k]
