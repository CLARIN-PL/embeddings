from abc import ABC
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, Final, List, Union

from embeddings.config.config_space import BaseConfigSpace, Parameter, SampledParameters
from embeddings.config.lightning_config import LightningConfigKeys
from embeddings.config.parameters import ConstantParameter, ParameterValues, SearchableParameter
from embeddings.data.io import T_path
from embeddings.utils.utils import read_yaml

DEFAULT_DEVICES = "auto"
DEFAULT_ACCELERATOR = "auto"


@dataclass
class LightingConfigSpace(BaseConfigSpace, LightningConfigKeys, ABC):
    embedding_name_or_path: InitVar[Union[T_path, List[T_path]]]
    devices: InitVar[Union[int, str, None, List[int]]] = field(default=DEFAULT_DEVICES)
    accelerator: InitVar[Union[str, None]] = field(default=DEFAULT_ACCELERATOR)

    param_embedding_name_or_path: Parameter = field(init=False)
    trainer_devices: Parameter = field(init=False)
    trainer_accelerator: Parameter = field(init=False)

    # Trainer parameters
    max_epochs: Parameter = SearchableParameter(
        name="max_epochs", type="int_uniform", low=1, high=30, step=1
    )
    # DataModule parameters
    mini_batch_size: Parameter = SearchableParameter(
        name="batch_size", type="log_int_uniform", low=8, high=64, step=1
    )
    max_seq_length: Parameter = ConstantParameter(
        name="max_seq_length",
        value=None,
    )
    # Task/LightningModule parameters
    optimizer: Parameter = SearchableParameter(
        name="optimizer",
        type="categorical",
        choices=["Adam", "AdamW"],
    )
    use_scheduler: Parameter = SearchableParameter(
        name="use_scheduler",
        type="categorical",
        choices=[False, True],
    )
    warmup_steps: Parameter = SearchableParameter(
        name="warmup_steps", type="int_uniform", low=0, high=200, step=10
    )
    learning_rate: Parameter = SearchableParameter(
        name="learning_rate", type="log_uniform", low=1e-6, high=1e-1
    )
    adam_epsilon: Parameter = SearchableParameter(
        name="adam_epsilon",
        type="uniform",
        low=0.0,
        high=1e-1,
    )
    weight_decay: Parameter = SearchableParameter(
        name="weight_decay",
        type="uniform",
        low=0.0,
        high=1e-1,
    )
    finetune_last_n_layers: Parameter = SearchableParameter(
        name="finetune_last_n_layers", type="categorical", choices=[-1, 0, 1, 3, 5, 7, 9]
    )
    classifier_dropout: Parameter = SearchableParameter(
        name="classifier_dropout", type="discrete_uniform", low=0.0, high=0.5, q=0.05
    )

    def __post_init__(
        self,
        embedding_name_or_path: Union[str, List[str]],
        devices: Union[int, str, None, List[int]],
        accelerator: Union[str, None],
    ) -> None:
        if isinstance(embedding_name_or_path, str):
            self.param_embedding_name_or_path: Parameter = ConstantParameter(
                name="embedding_name_or_path",
                value=embedding_name_or_path,
            )
        else:
            self.param_embedding_name_or_path: Parameter = SearchableParameter(
                name="embedding_name_or_path",
                type="categorical",
                choices=embedding_name_or_path,
            )

        self.trainer_devices = ConstantParameter(name="devices", value=devices)
        self.trainer_accelerator = ConstantParameter(name="accelerator", value=accelerator)

    @classmethod
    def parse_parameters(cls, parameters: Dict[str, ParameterValues]) -> SampledParameters:
        pipeline_kwargs = cls._pop_parameters(
            parameters=parameters, parameters_keys=cls.PIPELINE_KEYS
        )
        datamodule_kwargs = cls._pop_parameters(
            parameters=parameters, parameters_keys=cls.DATAMODULE_KEYS
        )
        task_model_kwargs = cls._pop_parameters(
            parameters=parameters, parameters_keys=cls.TASK_MODEL_KEYS
        )
        task_train_kwargs = cls._pop_parameters(
            parameters=parameters, parameters_keys=cls.TASK_TRAIN_KEYS
        )
        model_config_kwargs = cls._pop_parameters(
            parameters=parameters, parameters_keys=cls.MODEL_CONFIG_KEYS
        )

        batch_size = task_model_kwargs.pop("batch_size")
        task_model_kwargs["train_batch_size"] = task_model_kwargs["eval_batch_size"] = batch_size

        return {
            "datamodule_kwargs": datamodule_kwargs,
            "task_model_kwargs": task_model_kwargs,
            "task_train_kwargs": task_train_kwargs,
            "model_config_kwargs": model_config_kwargs,
            **pipeline_kwargs,
        }

    @classmethod
    def _parse_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        variables = {
            "embedding_name_or_path": config.pop("embedding_name_or_path"),
            "devices": config.pop("devices", DEFAULT_DEVICES),
            "accelerator": config.pop("accelerator", DEFAULT_ACCELERATOR),
        }
        parameters = cls._parse_config_params(config.pop("parameters"))
        cls._check_unmapped_parameters(config)
        return {**variables, **parameters}

    @classmethod
    def from_yaml(cls, path: T_path) -> "LightingConfigSpace":
        config = read_yaml(path)
        return cls(**cls._parse_config(config))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LightingConfigSpace":
        config = deepcopy(d)
        return cls(**cls._parse_config(config))


@dataclass
class LightingTextClassificationConfigSpace(LightingConfigSpace):
    @classmethod
    def parse_parameters(cls, parameters: Dict[str, ParameterValues]) -> SampledParameters:
        sampled_parameters = super().parse_parameters(parameters=parameters)
        cls._check_unmapped_parameters(parameters=parameters)
        return sampled_parameters


@dataclass
class LightingSequenceLabelingConfigSpace(LightingConfigSpace):
    label_all_tokens: Parameter = field(
        init=True, default=ConstantParameter(name="label_all_tokens", value=False)
    )

    @classmethod
    def parse_parameters(cls, parameters: Dict[str, ParameterValues]) -> SampledParameters:
        sampled_parameters = super().parse_parameters(parameters=parameters)
        extra_datamodule_keys: Final = {"label_all_tokens"}
        extra_datamodule_kwargs = BaseConfigSpace._pop_parameters(
            parameters=parameters, parameters_keys=extra_datamodule_keys
        )
        assert isinstance(sampled_parameters["datamodule_kwargs"], dict)
        sampled_parameters["datamodule_kwargs"].update(extra_datamodule_kwargs)
        cls._check_unmapped_parameters(parameters=parameters)
        return sampled_parameters
