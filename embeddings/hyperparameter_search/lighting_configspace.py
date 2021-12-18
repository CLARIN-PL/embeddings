from dataclasses import dataclass
from plistlib import Dict
from typing import Final

from embeddings.hyperparameter_search.configspace import (
    BaseConfigSpace,
    Parameter,
    SampledParameters,
)
from embeddings.hyperparameter_search.parameters import ConstantParameter, SearchableParameter
from embeddings.utils.utils import PrimitiveTypes


@dataclass
class LightingTextClassificationConfigSpace(BaseConfigSpace):
    max_epochs: Parameter = SearchableParameter(
        name="max_epochs",
        type="int_uniform",
        low=1, high=30
    )
    mini_batch_size: Parameter = SearchableParameter(
        name="batch_size",
        type="log_int_uniform",
        low=8, high=64
    )
    max_seq_length: Parameter = ConstantParameter(
        name="max_seq_length",
        value=512,
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
        type="log_uniform", low=1e-9, high=0,
    )
    weight_decay: Parameter = SearchableParameter(
        name="weight_decay",
        type="log_uniform", low=1e-9, high=0,
    )
    unfreeze_from: Parameter = SearchableParameter(
        name="unfreeze_from",
        type="categorical",
        choices=[-1, 4, 7, 9, 11, None],
    )

    @staticmethod
    def parse_parameters(parameters: Dict[str, PrimitiveTypes]) -> SampledParameters:
        dl_model_keys: Final = {"batch_size", "max_seq_length"}
        task_model_keys: Final = {
            "learning_rate",
            "unfreeze_from",
            "optimizer",
            "use_scheduler",
            "warmup_steps",
            "adam_epsilon",
            "weight_decay",
        }
        task_trainer_keys: Final = {
            "max_epochs",
        }
        dl_model_kwargs = BaseConfigSpace._pop_parameters(
            parameters=parameters, parameters_keys=dl_model_keys
        )
        task_model_kwargs = BaseConfigSpace._pop_parameters(
            parameters=parameters, parameters_keys=task_model_keys
        )
        task_trainer_kwargs = BaseConfigSpace._pop_parameters(
            parameters=parameters, parameters_keys=task_trainer_keys
        )
        task_model_kwargs["train_batch_size"] = dl_model_kwargs["batch_size"]
        task_model_kwargs["eval_batch_size"] = dl_model_kwargs["batch_size"]

        return {
            "dl_model_kwargs": dl_model_keys,
            "task_model_kwargs": task_model_kwargs,
            "task_trainer_kwargs": task_trainer_kwargs
        }
