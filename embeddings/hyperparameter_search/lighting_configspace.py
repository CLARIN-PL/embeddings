from dataclasses import InitVar, dataclass, field
from typing import Dict, Final, List, Union

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
class LightingTextClassificationConfigSpace(BaseConfigSpace):
    embedding_name: InitVar[Union[str, List[str]]]
    devices: InitVar[Union[int, str, None, List[int]]] = field(default="auto")
    accelerator: InitVar[Union[str, None]] = field(default="auto")

    param_embedding_name: Parameter = field(init=False)
    trainer_devices: Parameter = field(init=False)
    trainer_accelerator: Parameter = field(init=False)

    max_epochs: Parameter = SearchableParameter(
        name="max_epochs", type="int_uniform", low=1, high=30, step=1
    )
    mini_batch_size: Parameter = SearchableParameter(
        name="batch_size", type="log_int_uniform", low=8, high=64, step=1
    )
    max_seq_length: Parameter = ConstantParameter(
        name="max_seq_length",
        value=None,
    )
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

    @staticmethod
    def parse_parameters(parameters: Dict[str, ParameterValues]) -> SampledParameters:
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

        batch_size = pipeline_kwargs.pop("batch_size")
        pipeline_kwargs["train_batch_size"] = pipeline_kwargs["eval_batch_size"] = batch_size

        return {
            "datamodule_kwargs": datamodule_kwargs,
            "task_model_kwargs": task_model_kwargs,
            "task_train_kwargs": task_train_kwargs,
            **pipeline_kwargs,
        }
