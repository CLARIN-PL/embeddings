import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, TypeVar, Union

import optuna

from embeddings.hyperparameter_search.parameters import ConstantParameter, SearchableParameter

ConfigSpaceOutput = TypeVar("ConfigSpaceOutput")


class AbstractConfigSpace(abc.ABC):
    def _suggest_optuna_param(
        self, trial: optuna.trial.Trial, param_name: str
    ) -> Tuple[str, Union[None, float, int, str, bool]]:
        param: SearchableParameter = self.__getattribute__(param_name)
        return param.name, trial._suggest(name=param.name, distribution=param.distribution)

    def _map_parameters(
        self, trial: optuna.trial.Trial, parameters_types: Dict[str, Any]
    ) -> Dict[str, Union[None, float, int, str, bool]]:
        parameters: Dict[str, Union[None, float, int, str, bool]] = {}
        for param_name, param_type in parameters_types.items():
            if param_type is SearchableParameter:
                parameters.update([self._suggest_optuna_param(trial=trial, param_name=param_name)])

            elif param_type is ConstantParameter:
                const_param: ConstantParameter = self.__getattribute__(param_name)
                parameters[const_param.name] = const_param.value
            else:
                raise ValueError(
                    "ConfigSpace class should not contain other attributes other than SearchableParameter or ConstantParameter"
                )
        return parameters

    @abc.abstractmethod
    def sample_parameters(
        self, trial: optuna.trial.Trial
    ) -> Dict[str, Union[None, float, int, str, bool]]:
        pass


@dataclass
class SequenceLabelingConfigSpace(AbstractConfigSpace):
    hidden_size: SearchableParameter = SearchableParameter(
        name="hidden_size", type="int_uniform", low=128, high=2048, step=128
    )
    use_rnn: SearchableParameter = SearchableParameter(
        name="use_rnn", type="categorical", choices=[True, False]
    )
    rnn_type: SearchableParameter = SearchableParameter(
        name="rnn_type", type="categorical", choices=["LSTM", "GRU"]
    )
    rnn_layers: SearchableParameter = SearchableParameter(
        name="rnn_layers", type="int_uniform", low=1, high=3, step=1
    )
    dropout: SearchableParameter = SearchableParameter(
        name="dropout", type="discrete_uniform", low=0.0, high=0.5, q=0.05
    )
    word_dropout: SearchableParameter = SearchableParameter(
        name="word_dropout", type="discrete_uniform", low=0.0, high=0.5, q=0.05
    )
    locked_dropout: SearchableParameter = SearchableParameter(
        name="locked_dropout", type="discrete_uniform", low=0.0, high=0.5, q=0.05
    )
    reproject_embeddings: SearchableParameter = SearchableParameter(
        name="reproject_embeddings", type="categorical", choices=[True, False]
    )
    use_crf: SearchableParameter = SearchableParameter(
        name="use_crf", type="categorical", choices=[True, False]
    )
    learning_rate: SearchableParameter = SearchableParameter(
        name="learning_rate", type="log_uniform", low=1e-4, high=1e-1
    )
    mini_batch_size: SearchableParameter = SearchableParameter(
        name="mini_batch_size", type="log_int_uniform", low=16, high=256, step=1
    )
    max_epochs: SearchableParameter = SearchableParameter(
        name="max_epochs", type="int_uniform", low=1, high=5, step=1
    )
    param_selection_mode: ConstantParameter = field(
        init=False, default=ConstantParameter(name="param_selection_mode", value=True)
    )
    save_final_model: ConstantParameter = field(
        init=False, default=ConstantParameter(name="save_final_model", value=False)
    )

    def sample_parameters(
        self, trial: optuna.trial.Trial
    ) -> Dict[str, Union[None, float, int, str, bool]]:
        parameters = {}

        use_rnn_name, use_rnn_val = self._suggest_optuna_param(trial=trial, param_name="use_rnn")
        parameters[use_rnn_name] = use_rnn_val
        if use_rnn_val:
            for rnn_param in ("rnn_layers", "rnn_type"):
                parameters.update([self._suggest_optuna_param(trial=trial, param_name=rnn_param)])

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
