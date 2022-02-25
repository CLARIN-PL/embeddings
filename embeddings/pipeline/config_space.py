from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from embeddings.pipeline import kwargs_group
from embeddings.utils.loggers import get_logger

_logger = get_logger(__name__)


@dataclass
class ConfigSpace(ABC):
    @abstractmethod
    def from_yaml(self, yaml_filepath: Path) -> None:
        pass


@dataclass
class LightningConfigSpace(
    kwargs_group.LightningTaskTrainKwargs,
    kwargs_group.DatamoduleKwargs,
    kwargs_group.TaskModelKwargs,
    ConfigSpace,
):
    task_train_kwargs: dict = field(init=False, compare=False, default_factory=dict)
    datamodule_kwargs: dict = field(init=False, compare=False, default_factory=dict)
    task_model_kwargs: dict = field(init=False, compare=False, default_factory=dict)
    batch_encoding_kwargs: dict = None
    tokenizer_kwargs: dict = None
    load_dataset_kwargs: dict = None
    model_config_kwargs: dict = None
    early_stopping_kwargs: dict = None

    def __post_init__(self) -> None:
        self.parse_fields_to_kwargs(
            "task_train_kwargs", kwargs_group.LightningTaskTrainKwargs.__dataclass_fields__
        )
        self.parse_fields_to_kwargs(
            "datamodule_kwargs", kwargs_group.DatamoduleKwargs.__dataclass_fields__
        )
        self.parse_fields_to_kwargs(
            "task_model_kwargs", kwargs_group.TaskModelKwargs.__dataclass_fields__
        )

    def from_yaml(self, yaml_filepath: Path) -> None:
        with open(yaml_filepath, "r") as file:
            config_from_yaml = yaml.safe_load(file)

        self.update_params(config_from_yaml)

    def update_params(self, additional_params: dict) -> None:
        for param_name, param_value in additional_params.items():
            if not hasattr(self, param_name):
                _logger.warning(
                    "Parameter: {} do not belong to any group with parameters it will be assigned to other kwargs".format(
                        param_name
                    )
                )

            setattr(self, param_name, param_value)

        self.__post_init__()

    def update_specific_params_group(self, additional_params: dict, params_group_name: str) -> None:
        if not hasattr(self, params_group_name):
            _logger.error(
                "Unssupported group of params was selected: {}!".format(params_group_name)
            )

        if not getattr(self, params_group_name):
            setattr(self, params_group_name, {})

        for param_name, param_value in additional_params.items():
            getattr(self, params_group_name)[param_name] = param_value
