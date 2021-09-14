import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, TypeVar, Union

import optuna

from embeddings.hyperparameter_search.parameters import ConstantParameter, SearchableParameter

Parameter = Union[SearchableParameter, ConstantParameter]


class AbstractConfigSpace(abc.ABC):
    def _parse_parameter(
        self, param_name: str, trial: optuna.trial.Trial
    ) -> Tuple[str, Union[None, float, int, str, bool]]:
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
        self, trial: optuna.trial.Trial, parameters_types: Dict[str, Any]
    ) -> Dict[str, Union[None, float, int, str, bool]]:
        parameters: Dict[str, Union[None, float, int, str, bool]] = {}
        for param_name, param_type in parameters_types.items():
            parameters.update([self._parse_parameter(trial=trial, param_name=param_name)])

        return parameters

    @abc.abstractmethod
    def sample_parameters(
        self, trial: optuna.trial.Trial
    ) -> Dict[str, Union[None, float, int, str, bool]]:
        pass


@dataclass
class SequenceLabelingConfigSpace(AbstractConfigSpace):
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
    learning_rate: Parameter = SearchableParameter(
        name="learning_rate", type="log_uniform", low=1e-4, high=1e-1
    )
    mini_batch_size: Parameter = SearchableParameter(
        name="mini_batch_size", type="log_int_uniform", low=16, high=256, step=1
    )
    max_epochs: Parameter = SearchableParameter(
        name="max_epochs", type="int_uniform", low=1, high=5, step=1
    )
    param_selection_mode: Parameter = field(
        init=False, default=ConstantParameter(name="param_selection_mode", value=True)
    )
    save_final_model: Parameter = field(
        init=False, default=ConstantParameter(name="save_final_model", value=False)
    )

    def sample_parameters(
        self, trial: optuna.trial.Trial
    ) -> Dict[str, Union[None, float, int, str, bool]]:
        parameters = {}

        use_rnn_name, use_rnn_val = self._parse_parameter(trial=trial, param_name="use_rnn")
        parameters[use_rnn_name] = use_rnn_val

        if use_rnn_val:
            for rnn_param in ("rnn_layers", "rnn_type"):
                parameters.update([self._parse_parameter(trial=trial, param_name=rnn_param)])

        parameters.update(
            self._map_parameters(
                parameters_types={
                    param_name: param_cfg
                    for param_name, param_cfg in self.__annotations__.items()
                    if param_name not in {"rnn_layers", "rnn_type", "use_rnn"}
                },
                trial=trial,
            )
        )
        return parameters
