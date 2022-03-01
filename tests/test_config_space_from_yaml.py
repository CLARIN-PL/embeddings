from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml
from _pytest.tmpdir import TempdirFactory

from embeddings.config.optimized_config_space import OptimizedConfigSpace
from embeddings.config.optimized_flair_config_space import (
    OptimizedFlairModelTrainerConfigSpace,
    OptimizedFlairSequenceLabelingConfigSpace,
)
from embeddings.config.optimized_lighting_config_space import (
    OptimizedLightingTextClassificationConfigSpace,
)


@pytest.fixture(scope="module")
def tmp_path_module(tmpdir_factory: TempdirFactory) -> Path:
    path = tmpdir_factory.mktemp(__name__)
    return Path(path)


@pytest.fixture()
def base_config_dict() -> Dict[str, Any]:
    config = {
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
        }
    }
    return config


@pytest.fixture()
def flair_trainer_config_dict(base_config_dict: Dict[str, Any]) -> Dict[str, Any]:
    config = deepcopy(base_config_dict)
    config["embedding_name"] = "allegro/herbert-base-case"
    return config


@pytest.fixture()
def flair_sequence_labeling_config_dict(base_config_dict: Dict[str, Any]) -> Dict[str, Any]:
    config = deepcopy(base_config_dict)
    config["embedding_name"] = "allegro/herbert-base-case"
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


@pytest.fixture()
def lightning_classification_config_dict(base_config_dict: Dict[str, Any]) -> Dict[str, Any]:
    config = deepcopy(base_config_dict)
    config["embedding_name_or_path"] = "allegro/herbert-base-case"
    config["parameters"].update(
        {
            "max_seq_length": {
                "param_type": "constant",
                "name": "max_seq_length",
                "value": 128,
            },
            "optimizer": {
                "param_type": "searchable",
                "name": "optimizer",
                "type": "categorical",
                "choices": ["Adam", "AdamW"],
            },
            "use_scheduler": {
                "param_type": "searchable",
                "name": "use_scheduler",
                "type": "categorical",
                "choices": [False, True],
            },
            "warmup_steps": {
                "param_type": "searchable",
                "name": "warmup_steps",
                "type": "int_uniform",
                "low": 0,
                "high": 200,
                "step": 10,
            },
            "adam_epsilon": {
                "param_type": "searchable",
                "name": "adam_epsilon",
                "type": "uniform",
                "low": 0.0,
                "high": 0.1,
            },
            "weight_decay": {
                "param_type": "searchable",
                "name": "weight_decay",
                "type": "uniform",
                "low": 0.0,
                "high": 0.1,
            },
            "finetune_last_n_layers": {
                "param_type": "searchable",
                "name": "finetune_last_n_layers",
                "type": "categorical",
                "choices": [-1, 0, 1, 3, 5, 7, 9],
            },
            "classifier_dropout": {
                "param_type": "searchable",
                "name": "classifier_dropout",
                "type": "discrete_uniform",
                "low": 0.0,
                "high": 0.5,
                "q": 0.05,
            },
        }
    )
    return config


@pytest.fixture()
def lightning_classification_wrong_param_config_dict(
    lightning_classification_config_dict: Dict[str, Any]
) -> Dict[str, Any]:
    config = deepcopy(lightning_classification_config_dict)
    config["parameters"].update(
        {"label_all_tokens": {"param_type": "constant", "name": "label_all_tokens", "value": False}}
    )
    return config


@pytest.fixture()
def base_yaml_config_file_path(tmp_path_module: Path, base_config_dict: Dict[str, Any]) -> Path:
    output_path = tmp_path_module.joinpath("config.yaml")
    with open(output_path, "w") as f:
        yaml.dump(base_config_dict, f, default_flow_style=False)
    return output_path


@pytest.fixture()
def flair_trainer_yaml_config_file_path(
    tmp_path_module: Path, flair_trainer_config_dict: Dict[str, Any]
) -> Path:
    output_path = tmp_path_module.joinpath("flair_trainer_config.yaml")
    with open(output_path, "w") as f:
        yaml.dump(flair_trainer_config_dict, f, default_flow_style=False)
    return output_path


@pytest.fixture()
def flair_sequence_labeling_yaml_config_file_path(
    tmp_path_module: Path, flair_sequence_labeling_config_dict: Dict[str, Any]
) -> Path:
    output_path = tmp_path_module.joinpath("flair_sequence_labeling_config.yaml")
    with open(output_path, "w") as f:
        yaml.dump(flair_sequence_labeling_config_dict, f, default_flow_style=False)
    return output_path


@pytest.fixture()
def lightning_classification_yaml_config_file_path(
    tmp_path_module: Path, lightning_classification_config_dict: Dict[str, Any]
) -> Path:
    output_path = tmp_path_module.joinpath("lightning_classification_config.yaml")
    with open(output_path, "w") as f:
        yaml.dump(lightning_classification_config_dict, f, default_flow_style=False)
    return output_path


@pytest.fixture()
def lightning_classification_wrong_param_yaml_config_file_path(
    tmp_path_module: Path,
    lightning_classification_wrong_param_config_dict: Dict[str, Any],
) -> Path:
    output_path = tmp_path_module.joinpath("lightning_classification_no_embedding_name_config.yaml")
    with open(output_path, "w") as f:
        yaml.dump(lightning_classification_wrong_param_config_dict, f, default_flow_style=False)
    return output_path


def assert_config_with_yaml(config_space: OptimizedConfigSpace, config: Dict[str, Any]) -> None:
    cs_params = config.pop("parameters")
    for param_name, param_values in cs_params.items():
        param_values.pop("param_type")
        for attr_name, attr_value in param_values.items():
            if attr_name != "param_type":
                assert (
                    getattr(getattr(config_space, param_name), attr_name)
                    == cs_params[param_name][attr_name]
                )


def test_flair_trainer_from_yaml_config(
    flair_trainer_config_dict: Dict[str, Any], flair_trainer_yaml_config_file_path: Path
) -> None:
    flair_trainer_config_space = OptimizedFlairModelTrainerConfigSpace.from_yaml(
        flair_trainer_yaml_config_file_path
    )
    assert hasattr(flair_trainer_config_space, "param_embedding_name")
    assert (
        flair_trainer_config_space.param_embedding_name.value
        == flair_trainer_config_dict["embedding_name"]
    )
    assert_config_with_yaml(flair_trainer_config_space, flair_trainer_config_dict)


def test_flair_sequence_labeling_from_yaml_trainer_config(
    flair_trainer_config_dict: Dict[str, Any], flair_trainer_yaml_config_file_path: Path
) -> None:
    flair_trainer_config_space = OptimizedFlairSequenceLabelingConfigSpace.from_yaml(
        flair_trainer_yaml_config_file_path
    )
    assert hasattr(flair_trainer_config_space, "param_embedding_name")
    assert (
        flair_trainer_config_space.param_embedding_name.value
        == flair_trainer_config_dict["embedding_name"]
    )
    assert_config_with_yaml(flair_trainer_config_space, flair_trainer_config_dict)


def test_flair_sequence_labeling_from_yaml_specific_config(
    flair_sequence_labeling_config_dict: Dict[str, Any],
    flair_sequence_labeling_yaml_config_file_path: Path,
) -> None:
    flair_sequence_labeling_config_space = OptimizedFlairSequenceLabelingConfigSpace.from_yaml(
        flair_sequence_labeling_yaml_config_file_path
    )
    assert hasattr(flair_sequence_labeling_config_space, "param_embedding_name")
    assert (
        flair_sequence_labeling_config_space.param_embedding_name.value
        == flair_sequence_labeling_config_dict["embedding_name"]
    )
    assert_config_with_yaml(
        flair_sequence_labeling_config_space, flair_sequence_labeling_config_dict
    )


def test_lightning_classification_from_yaml_config(
    lightning_classification_config_dict: Dict[str, Any],
    lightning_classification_yaml_config_file_path: Path,
) -> None:
    lightning_classification_config_space = (
        OptimizedLightingTextClassificationConfigSpace.from_yaml(
            lightning_classification_yaml_config_file_path
        )
    )
    assert hasattr(lightning_classification_config_space, "param_embedding_name_or_path")
    assert (
        lightning_classification_config_space.param_embedding_name_or_path.value
        == lightning_classification_config_dict["embedding_name_or_path"]
    )
    assert_config_with_yaml(
        lightning_classification_config_space, lightning_classification_config_dict
    )


def test_wrong_config_given(
    lightning_classification_yaml_config_file_path: Path,
    flair_sequence_labeling_yaml_config_file_path: Path,
    flair_trainer_yaml_config_file_path: Path,
):
    with pytest.raises(KeyError):
        OptimizedLightingTextClassificationConfigSpace.from_yaml(
            flair_sequence_labeling_yaml_config_file_path
        )
    with pytest.raises(KeyError):
        OptimizedLightingTextClassificationConfigSpace.from_yaml(
            flair_trainer_yaml_config_file_path
        )
    with pytest.raises(KeyError):
        OptimizedFlairSequenceLabelingConfigSpace.from_yaml(
            lightning_classification_yaml_config_file_path
        )


def test_no_embedding_name_given(
    base_yaml_config_file_path: Path,
):
    with pytest.raises(KeyError):
        OptimizedLightingTextClassificationConfigSpace.from_yaml(base_yaml_config_file_path)
    with pytest.raises(KeyError):
        OptimizedFlairSequenceLabelingConfigSpace.from_yaml(base_yaml_config_file_path)


def test_wrong_param_given(
    lightning_classification_wrong_param_yaml_config_file_path: Path,
):
    with pytest.raises(TypeError):
        OptimizedLightingTextClassificationConfigSpace.from_yaml(
            lightning_classification_wrong_param_yaml_config_file_path
        )


def test_lightning_classification_from_dict_config(
    lightning_classification_config_dict: Dict[str, Any],
    lightning_classification_yaml_config_file_path: Path,
) -> None:
    config_space_from_yaml = OptimizedLightingTextClassificationConfigSpace.from_yaml(
        lightning_classification_yaml_config_file_path
    )
    config_space_from_dict = OptimizedLightingTextClassificationConfigSpace.from_dict(
        lightning_classification_config_dict
    )
    config_space_attributes_from_yaml = list(config_space_from_yaml._get_fields().keys())
    config_space_attributes_from_dict = list(config_space_from_dict._get_fields().keys())
    assert config_space_attributes_from_yaml == config_space_attributes_from_dict

    for attr_key in config_space_attributes_from_yaml:
        assert (
            getattr(config_space_from_dict, attr_key).value
            == getattr(config_space_from_dict, attr_key).value
        )
