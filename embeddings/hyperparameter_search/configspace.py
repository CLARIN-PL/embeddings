import abc
import dataclasses
import functools
from abc import ABC
from typing import Any, Dict, List, Set, Tuple, Type, TypeVar, Union

import optuna

from embeddings.data.io import T_path
from embeddings.embedding.auto_flair import AutoFlairWordEmbedding
from embeddings.embedding.flair_embedding import FlairTransformerEmbedding
from embeddings.embedding.static.embedding import StaticEmbedding
from embeddings.hyperparameter_search.parameters import (
    ConstantParameter,
    ParameterValues,
    SearchableParameter,
)

Parameter = Union[SearchableParameter, ConstantParameter]
ParsedParameters = TypeVar("ParsedParameters")
SampledParameters = Dict[str, Union[ParameterValues, Dict[str, ParameterValues]]]
ConfigSpace = TypeVar("ConfigSpace", bound="BaseConfigSpace")


class BaseConfigSpace(ABC):
    def _parse_parameter(
        self, param_name: str, trial: optuna.trial.Trial
    ) -> Tuple[str, ParameterValues]:
        param: Parameter = self.__getattribute__(param_name)
        if isinstance(param, SearchableParameter):
            param.value = trial._suggest(name=param.name, distribution=param.distribution)
            return param.name, param.value
        elif isinstance(param, ConstantParameter):
            trial.set_user_attr(param.name, param.value)
            return param.name, param.value
        else:
            raise ValueError(
                f"Parameter type {type(param)} is not suported! "
                f"Supported types are: {SearchableParameter.__name__} and {ConstantParameter.__name__}"
            )

    def _map_parameters(
        self, trial: optuna.trial.Trial, parameters_names: List[str]
    ) -> Dict[str, ParameterValues]:
        parameters: Dict[str, ParameterValues] = {}
        for param_name in parameters_names:
            parameters.update([self._parse_parameter(trial=trial, param_name=param_name)])

        return parameters

    def _map_task_specific_parameters(
        self, trial: optuna.trial.Trial
    ) -> Tuple[Dict[str, ParameterValues], Set[str]]:
        return dict(), set()

    @classmethod
    def _get_fields(cls) -> Dict[str, Type[Parameter]]:
        return {field_.name: field_.type for field_ in dataclasses.fields(cls)}

    def sample_parameters(self, trial: optuna.trial.Trial) -> Dict[str, ParameterValues]:
        task_params, mapped_params_names = self._map_task_specific_parameters(trial)
        params = self._map_parameters(
            parameters_names=[
                param_name
                for param_name in self._get_fields().keys()
                if param_name not in mapped_params_names
            ],
            trial=trial,
        )
        self._check_duplicated_parameters(params, task_params)
        return {**params, **task_params}

    @staticmethod
    def _pop_parameters(
        parameters: Dict[str, ParameterValues], parameters_keys: Set[str]
    ) -> Dict[str, ParameterValues]:
        return {k: parameters.pop(k) for k in parameters_keys if k in parameters}

    @staticmethod
    def _check_duplicated_parameters(*parameter_dicts: Any) -> Any:
        assert not functools.reduce(set.intersection, (set(d.keys()) for d in parameter_dicts))

    @staticmethod
    def _check_unmapped_parameters(parameters: Dict[str, ParameterValues]) -> None:
        if len(parameters):
            raise ValueError(
                f"Some of the parameters are not mapped. Unmapped parameters: {parameters}"
            )

    # classmethod instead of staticmethod omit the mypy error: Argument 2 for "super" not an instance of argument 1 (https://github.com/python/mypy/issues/9282)
    @classmethod
    @abc.abstractmethod
    def parse_parameters(cls, parameters: Dict[str, ParameterValues]) -> SampledParameters:
        pass

    @staticmethod
    def _retrieve_embedding_type(embedding_name: str) -> str:
        embedding = AutoFlairWordEmbedding.from_hub(repo_id=embedding_name)
        if isinstance(embedding, FlairTransformerEmbedding):
            embedding_type = "dynamic"
        elif isinstance(embedding, StaticEmbedding):
            embedding_type = "static"
        else:
            raise TypeError("Embedding type not recognized")
        return embedding_type

    @staticmethod
    def _parse_config_params(parameters: Dict[str, Any]) -> Dict[str, Parameter]:
        parsed_parameters = {}
        for param_key, param_values in parameters.items():
            param_type = param_values.pop("param_type")
            param: Parameter
            if param_type == "constant":
                param = ConstantParameter(**param_values)
            elif param_type == "searchable":
                param = SearchableParameter(**param_values)
            else:
                raise ValueError(f"Unrecognized parameter type for parameter {param_key}")
            parsed_parameters.update({param_key: param})
        return parsed_parameters

    @classmethod
    @abc.abstractmethod
    def _parse_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @classmethod
    @abc.abstractmethod
    def from_yaml(cls, path: T_path) -> "BaseConfigSpace":
        pass

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BaseConfigSpace":
        pass
