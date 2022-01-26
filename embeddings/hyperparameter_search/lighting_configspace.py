from dataclasses import InitVar, dataclass, field
from typing import Dict, Final, List, Union

import yaml

from embeddings.defaults import LIGHTNING_CONFIG_PATH
from embeddings.hyperparameter_search.configspace import (
    BaseConfigSpace,
    Parameter,
    SampledParameters,
)
from embeddings.hyperparameter_search.parameters import (
    ConstantParameter,
    ParameterValues,
    SearchableParameter,
)


@dataclass
class LightingConfigSpace(BaseConfigSpace):
    embedding_name: InitVar[Union[str, List[str]]]
    devices: InitVar[Union[int, str, None, List[int]]] = field(default="auto")
    accelerator: InitVar[Union[str, None]] = field(default="auto")

    param_embedding_name: Parameter = field(init=False)
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
        embedding_name: Union[str, List[str]],
        devices: Union[int, str, None, List[int]],
        accelerator: Union[str, None],
    ) -> None:
        if isinstance(embedding_name, str):
            self.param_embedding_name: Parameter = ConstantParameter(
                name="embedding_name",
                value=embedding_name,
            )
        else:
            self.param_embedding_name: Parameter = SearchableParameter(
                name="embedding_name",
                type="categorical",
                choices=embedding_name,
            )

        self.trainer_devices = ConstantParameter(name="devices", value=devices)
        self.trainer_accelerator = ConstantParameter(name="accelerator", value=accelerator)

    @classmethod
    def parse_parameters(cls, parameters: Dict[str, ParameterValues]) -> SampledParameters:
        pipeline_keys: Final = {"batch_size", "finetune_last_n_layers", "embedding_name"}
        datamodule_keys: Final = {"max_seq_length"}
        task_model_keys: Final = {
            "learning_rate",
            "optimizer",
            "use_scheduler",
            "warmup_steps",
            "adam_epsilon",
            "weight_decay",
        }
        task_train_keys: Final = {"max_epochs", "devices", "accelerator"}
        model_config_keys: Final = {"classifier_dropout"}

        pipeline_kwargs = BaseConfigSpace._pop_parameters(
            parameters=parameters, parameters_keys=pipeline_keys
        )
        datamodule_kwargs = BaseConfigSpace._pop_parameters(
            parameters=parameters, parameters_keys=datamodule_keys
        )
        task_model_kwargs = BaseConfigSpace._pop_parameters(
            parameters=parameters, parameters_keys=task_model_keys
        )
        task_train_kwargs = BaseConfigSpace._pop_parameters(
            parameters=parameters, parameters_keys=task_train_keys
        )
        model_config_kwargs = BaseConfigSpace._pop_parameters(
            parameters=parameters, parameters_keys=model_config_keys
        )

        batch_size = pipeline_kwargs.pop("batch_size")
        pipeline_kwargs["train_batch_size"] = pipeline_kwargs["eval_batch_size"] = batch_size

        return {
            "datamodule_kwargs": datamodule_kwargs,
            "task_model_kwargs": task_model_kwargs,
            "task_train_kwargs": task_train_kwargs,
            "model_config_kwargs": model_config_kwargs,
            **pipeline_kwargs,
        }

    @classmethod
    def read_from_yaml(cls) -> "LightingConfigSpace":
        with open(LIGHTNING_CONFIG_PATH, 'r') as file:
            data = yaml.load(file, Loader=yaml.CLoader)
            data = data["LightingConfigSpace"]

        return cls(
            embedding_name=data["embedding_name"],
            devices=field(**data["devices"]),
            accelerator=field(**data["accelerator"]),
            max_epochs=SearchableParameter(**data["max_epochs"]),
            mini_batch_size=SearchableParameter(**data["mini_batch_size"]),
            max_seq_length=ConstantParameter(**data["max_seq_length"]),
            optimizer=SearchableParameter(**data["optimizer"]),
            use_scheduler=SearchableParameter(**data["use_scheduler"]),
            warmup_steps=SearchableParameter(**data["warmup_steps"]),
            learning_rate=SearchableParameter(**data["learning_rate"]),
            adam_epsilon=SearchableParameter(**data["adam_epsilon"]),
            weight_decay=SearchableParameter(**data["weight_decay"]),
            finetune_last_n_layers=SearchableParameter(**data["finetune_last_n_layers"]),
            classifier_dropout=SearchableParameter(**data["classifier_dropout"]),
        )


@dataclass
class LightingTextClassificationConfigSpace(LightingConfigSpace):
    pass


@dataclass
class LightingSequenceLabelingConfigSpace(LightingConfigSpace):
    label_all_tokens: Parameter = field(
        init=False, default=ConstantParameter(name="label_all_tokens", value=False)
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

        return sampled_parameters
