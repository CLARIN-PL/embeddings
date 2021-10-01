import abc
from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, Final, List, Set, Tuple, Type, TypeVar, Union

import optuna

from embeddings.hyperparameter_search.parameters import ConstantParameter, SearchableParameter
from embeddings.utils.utils import PrimitiveTypes

Parameter = Union[SearchableParameter, ConstantParameter]
ParsedParameters = TypeVar("ParsedParameters")
SampledParameters = Dict[str, Union[PrimitiveTypes, Dict[str, PrimitiveTypes]]]


class ConfigSpace(ABC):
    def _parse_parameter(
        self, param_name: str, trial: optuna.trial.Trial
    ) -> Tuple[str, PrimitiveTypes]:
        param: Parameter = self.__getattribute__(param_name)
        if isinstance(param, SearchableParameter):
            return param.name, trial._suggest(name=param.name, distribution=param.distribution)
        elif isinstance(param, ConstantParameter):
            return param.name, param.value
        else:
            raise ValueError(
                f"Parameter type {type(param)} is not suported! "
                "Supported types are: SearchableParameter and ConstantParameter"
            )

    def _map_parameters(
        self, trial: optuna.trial.Trial, parameters_names: List[str]
    ) -> Dict[str, PrimitiveTypes]:
        parameters: Dict[str, PrimitiveTypes] = {}
        for param_name in parameters_names:
            parameters.update([self._parse_parameter(trial=trial, param_name=param_name)])

        return parameters

    def _map_task_specific_parameters(
        self, trial: optuna.trial.Trial
    ) -> Tuple[Dict[str, PrimitiveTypes], Set[str]]:
        return dict(), set()

    @classmethod
    def _get_annotations(cls) -> Dict[str, Type[Parameter]]:
        annotations = {}
        for c in cls.mro():
            annotations.update(getattr(c, "__annotations__", {}))
        return annotations

    def sample_parameters(self, trial: optuna.trial.Trial) -> Dict[str, PrimitiveTypes]:
        task_params, mapped_params_names = self._map_task_specific_parameters(trial)
        params = self._map_parameters(
            parameters_names=[
                param_name
                for param_name in self._get_annotations().keys()
                if param_name not in mapped_params_names
            ],
            trial=trial,
        )
        return {**params, **task_params}

    @staticmethod
    @abc.abstractmethod
    def parse_parameters(parameters: Dict[str, PrimitiveTypes]) -> SampledParameters:
        pass


@dataclass  # type: ignore
class AbstractFlairModelTrainerConfigSpace(ConfigSpace, ABC):
    param_selection_mode: Parameter = field(
        init=False, default=ConstantParameter(name="param_selection_mode", value=True)
    )
    save_final_model: Parameter = field(
        init=False, default=ConstantParameter(name="save_final_model", value=False)
    )
    learning_rate: Parameter = SearchableParameter(
        name="learning_rate", type="log_uniform", low=1e-4, high=1e-1
    )
    mini_batch_size: Parameter = SearchableParameter(
        name="mini_batch_size", type="log_int_uniform", low=16, high=256, step=1
    )
    max_epochs: Parameter = SearchableParameter(
        name="max_epochs", type="int_uniform", low=1, high=5, step=1
    )

    @staticmethod
    def _parse_model_trainer_parameters(
        parameters: Dict[str, PrimitiveTypes]
    ) -> Tuple[Dict[str, PrimitiveTypes], Dict[str, PrimitiveTypes]]:
        task_train_keys: Final = {
            "learning_rate",
            "mini_batch_size",
            "max_epochs",
            "param_selection_mode",
            "save_final_model",
        }
        task_train_kwargs = {k: parameters.pop(k) for k in task_train_keys if k in parameters}
        return parameters, task_train_kwargs


class FlairModelTrainerConfigSpace(AbstractFlairModelTrainerConfigSpace):
    @staticmethod
    def parse_parameters(parameters: Dict[str, PrimitiveTypes]) -> SampledParameters:
        (
            parameters,
            task_train_kwargs,
        ) = FlairModelTrainerConfigSpace._parse_model_trainer_parameters(parameters)
        if len(parameters):
            raise ValueError(
                f"Some of the parameters are not mapped. Unmapped parameters: {parameters}"
            )
        return {"task_train_kwargs": task_train_kwargs}


@dataclass
class SequenceLabelingConfigSpace(AbstractFlairModelTrainerConfigSpace):
    hidden_size: Parameter = SearchableParameter(
        name="hidden_size", type="int_uniform", low=128, high=2048, step=128
    )
    use_rnn: Parameter = SearchableParameter(
        name="use_rnn", type="categorical", choices=[True, False]
    )
    rnn_type: Parameter = SearchableParameter(
        name="rnn_type", type="categorical", choices=["LSTM", "GRU"]
    )
    rnn_layers: Parameter = SearchableParameter(
        name="rnn_layers", type="int_uniform", low=1, high=3, step=1
    )
    dropout: Parameter = SearchableParameter(
        name="dropout", type="discrete_uniform", low=0.0, high=0.5, q=0.05
    )
    word_dropout: Parameter = SearchableParameter(
        name="word_dropout", type="discrete_uniform", low=0.0, high=0.5, q=0.05
    )
    locked_dropout: Parameter = SearchableParameter(
        name="locked_dropout", type="discrete_uniform", low=0.0, high=0.5, q=0.05
    )
    reproject_embeddings: Parameter = SearchableParameter(
        name="reproject_embeddings", type="categorical", choices=[True, False]
    )
    use_crf: Parameter = SearchableParameter(
        name="use_crf", type="categorical", choices=[True, False]
    )

    def _map_task_specific_parameters(
        self, trial: optuna.trial.Trial
    ) -> Tuple[Dict[str, PrimitiveTypes], Set[str]]:
        parameters = {}
        use_rnn_name, use_rnn_val = self._parse_parameter(trial=trial, param_name="use_rnn")
        parameters[use_rnn_name] = use_rnn_val

        if use_rnn_val:
            for rnn_param in ("rnn_layers", "rnn_type"):
                parameters.update([self._parse_parameter(trial=trial, param_name=rnn_param)])

        mapped_parameters: Final[Set[str]] = {"rnn_layers", "rnn_type", "use_rnn"}
        return parameters, mapped_parameters

    @staticmethod
    def parse_parameters(parameters: Dict[str, PrimitiveTypes]) -> SampledParameters:
        hidden_size = parameters.pop("hidden_size")
        assert isinstance(hidden_size, int)
        task_model_keys: Final = {
            "use_rnn",
            "dropout",
            "word_dropout",
            "locked_dropout",
            "reproject_embeddings",
            "use_crf",
            "rnn_layers",
            "rnn_type",
        }

        task_model_kwargs = {k: parameters.pop(k) for k in task_model_keys if k in parameters}
        parameters, task_train_kwargs = SequenceLabelingConfigSpace._parse_model_trainer_parameters(
            parameters=parameters
        )

        if len(parameters):
            raise ValueError(
                f"Some of the parameters are not mapped. Unmapped parameters: {parameters}"
            )
        return {
            "hidden_size": hidden_size,
            "task_model_kwargs": task_model_kwargs,
            "task_train_kwargs": task_train_kwargs,
        }


CS = TypeVar("CS", bound=ConfigSpace)
