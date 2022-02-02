from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict

import pytest
import yaml

from embeddings.hyperparameter_search.configspace import BaseConfigSpace
from embeddings.hyperparameter_search.flair_configspace import FlairModelTrainerConfigSpace
from embeddings.hyperparameter_search.flair_configspace import (
    SequenceLabelingConfigSpace as FlairSequenceLabelingConfigSpace,
)


@pytest.fixture(scope="module")
def output_path() -> "TemporaryDirectory[str]":
    return TemporaryDirectory()


@pytest.fixture(scope="module")
def flair_trainer_config() -> Dict[str, Any]:
    config = {
        "embedding_name": "herbert-base-cased",
        "parameters": {
            "learning_rate": {
                "high": 0.1,
                "low": 0.0001,
                "name": "learning_rate",
                "param_type": "searchable",
                "type": "log_uniform",
            },
            "max_epochs": {
                "high": 5,
                "low": 1,
                "name": "max_epochs",
                "param_type": "searchable",
                "step": 1,
                "type": "int_uniform",
            },
            "mini_batch_size": {
                "high": 256,
                "low": 16,
                "name": "mini_batch_size",
                "param_type": "searchable",
                "step": 1,
                "type": "log_int_uniform",
            },
        },
    }
    return config


@pytest.fixture(scope="module")
def flair_sequence_labeling_config(flair_trainer_config: Dict[str, Any]) -> Dict[str, Any]:
    config = flair_trainer_config
    config["parameters"].update(
        {
            "hidden_size": {
                "param_type": "searchable",
                "name": "hidden_size",
                "type": "int_uniform",
                "low": 128,
                "high": 2048,
                "step": 128,
            },
            "use_rnn": {
                "param_type": "searchable",
                "name": "use_rnn",
                "type": "categorical",
                "choices": [True, False],
            },
            "rnn_type": {
                "param_type": "searchable",
                "name": "rnn_type",
                "type": "categorical",
                "choices": ["LSTM", "GRU"],
            },
            "rnn_layers": {
                "param_type": "searchable",
                "name": "rnn_layers",
                "type": "int_uniform",
                "low": 1,
                "high": 3,
                "step": 1,
            },
            "dropout": {
                "param_type": "searchable",
                "name": "dropout",
                "type": "discrete_uniform",
                "low": 0.0,
                "high": 0.5,
                "q": 0.05,
            },
            "locked_dropout": {
                "param_type": "searchable",
                "name": "locked_dropout",
                "type": "discrete_uniform",
                "low": 0.0,
                "high": 0.5,
                "q": 0.05,
            },
            "reproject_embeddings": {
                "param_type": "searchable",
                "name": "reproject_embeddings",
                "type": "categorical",
                "choices": [True, False],
            },
            "use_crf": {
                "param_type": "searchable",
                "name": "use_crf",
                "type": "categorical",
                "choices": [True, False],
            },
        }
    )
    return config


@pytest.fixture(scope="module")
def flair_trainer_yaml_config_file_path(
    output_path: "TemporaryDirectory[str]", flair_trainer_config: Dict[str, Any]
) -> Path:
    output_path = Path(output_path.name).joinpath("config.yml")
    with open(output_path, "w") as f:
        yaml.dump(flair_trainer_config, f, default_flow_style=False)
    return output_path


@pytest.fixture(scope="module")
def flair_sequence_labeling_yaml_config_file_path(
    output_path: "TemporaryDirectory[str]", flair_sequence_labeling_config: Dict[str, Any]
) -> Path:
    output_path = Path(output_path.name).joinpath("config.yml")
    with open(output_path, "w") as f:
        yaml.dump(flair_sequence_labeling_config, f, default_flow_style=False)
    return output_path


def compare_config_with_yaml(config_space: BaseConfigSpace, config: Dict[str, Any]) -> None:
    assert hasattr(config_space, "param_embedding_name")
    assert config_space.param_embedding_name.value == config["embedding_name"]
    cs_params = config.pop("parameters")
    for param_name, param_values in cs_params.items():
        param_values.pop("param_type")
        for attr_name, attr_value in param_values.items():
            if attr_name != "param_type":
                assert (
                    getattr(getattr(config_space, param_name), attr_name)
                    == cs_params[param_name][attr_name]
                )


def test_flair_trainer_yaml_config(
    flair_trainer_config: Dict[str, Any], flair_trainer_yaml_config_file_path: Path
) -> None:
    flair_trainer_config_space = FlairModelTrainerConfigSpace.from_yaml(
        flair_trainer_yaml_config_file_path
    )
    compare_config_with_yaml(flair_trainer_config_space, flair_trainer_config)


def test_flair_sequence_labeling_yaml_trainer_config(
    flair_trainer_config: Dict[str, Any], flair_trainer_yaml_config_file_path: Path
) -> None:
    flair_trainer_config_space = FlairSequenceLabelingConfigSpace.from_yaml(
        flair_trainer_yaml_config_file_path
    )
    compare_config_with_yaml(flair_trainer_config_space, flair_trainer_config)


def test_flair_sequence_labeling_yaml_specific_config(
    flair_sequence_labeling_config: Dict[str, Any],
    flair_sequence_labeling_yaml_config_file_path: Path,
) -> None:
    flair_sequence_labeling_config_space = FlairSequenceLabelingConfigSpace.from_yaml(
        flair_sequence_labeling_yaml_config_file_path
    )
    compare_config_with_yaml(flair_sequence_labeling_config_space, flair_sequence_labeling_config)
