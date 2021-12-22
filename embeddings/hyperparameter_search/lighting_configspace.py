from dataclasses import InitVar, dataclass, field
from typing import Dict, Final, List, Union

from embeddings.hyperparameter_search.configspace import (
    BaseConfigSpace,
    Parameter,
    SampledParameters,
)
from embeddings.hyperparameter_search.parameters import ConstantParameter, SearchableParameter
from embeddings.utils.utils import PrimitiveTypes


@dataclass
class LightingTextClassificationConfigSpace(BaseConfigSpace):
    embedding_name: InitVar[Union[str, List[str]]]
    param_embedding_name: Parameter = field(init=False)
    max_epochs: Parameter = SearchableParameter(
        name="max_epochs", type="int_uniform", low=1, high=30
    )
    mini_batch_size: Parameter = SearchableParameter(
        name="batch_size", type="log_int_uniform", low=8, high=64
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
        type="log_uniform",
        low=1e-9,
        high=0,
    )
    weight_decay: Parameter = SearchableParameter(
        name="weight_decay",
        type="log_uniform",
        low=1e-9,
        high=0,
    )
    finetune_last_n_layers: Parameter = SearchableParameter(
        name="finetune_last_n_layers",
        type="categorical",
        choices=[-1, 4, 7, 9, 11, None],
    )

    def __post_init__(self, embedding_name: Union[str, List[str]]) -> None:
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

    @staticmethod
    def parse_parameters(parameters: Dict[str, PrimitiveTypes]) -> SampledParameters:
        pipeline_keys: Final = {"batch_size", "unfreeze_from", "embedding_name"}
        datamodule_keys: Final = {"max_seq_length"}
        task_model_keys: Final = {
            "learning_rate",
            "optimizer",
            "use_scheduler",
            "warmup_steps",
            "adam_epsilon",
            "weight_decay",
        }
        task_trainer_keys: Final = {
            "max_epochs",
        }
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
            parameters=parameters, parameters_keys=task_trainer_keys
        )

        batch_size = pipeline_kwargs.pop("batch_size")
        pipeline_kwargs["train_batch_size"] = batch_size
        pipeline_kwargs["eval_batch_size"] = batch_size

        return {
            "datamodule_kwargs": datamodule_kwargs,
            "task_model_kwargs": task_model_kwargs,
            "task_train_kwargs": task_train_kwargs,
            **pipeline_kwargs,
        }
