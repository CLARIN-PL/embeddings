import abc
from abc import ABC
from dataclasses import dataclass, field
from typing import Generic, Literal, Optional, Sequence, TypeVar, Union, get_args

import optuna
import optuna.distributions

from embeddings.utils.utils import Numeric, PrimitiveTypes

Distribution = TypeVar("Distribution")
ParameterTypes = Literal[
    "categorical",
    "uniform",
    "log_uniform",
    "discrete_uniform",
    "int_uniform",
    "log_int_uniform",
]


@dataclass(frozen=True)
class ConstantParameter:
    name: str
    value: PrimitiveTypes


# Type: ignore is due to https://github.com/python/mypy/issues/5374
@dataclass  # type:ignore
class AbstractSearchableParameter(ABC, Generic[Distribution]):
    name: str
    type: ParameterTypes
    distribution: Distribution = field(init=False)
    low: Optional[Numeric] = None
    high: Optional[Numeric] = None
    step: Optional[Numeric] = None
    q: Optional[float] = None
    choices: Optional[Sequence[PrimitiveTypes]] = None

    @staticmethod
    def _check_additional_params_passed(
        variables: Sequence[Union[PrimitiveTypes, Sequence[PrimitiveTypes]]]
    ) -> None:
        for var in variables:
            assert var is None

    def __post_init__(self) -> None:
        if self.type == "categorical":
            assert self.choices is not None
            AbstractSearchableParameter._check_additional_params_passed(
                (self.low, self.high, self.step, self.q)
            )

            self.distribution = self.get_categorical_distribution(choices=self.choices)
        elif self.type == "uniform":
            assert isinstance(self.low, float)
            assert isinstance(self.high, float)
            AbstractSearchableParameter._check_additional_params_passed(
                (self.choices, self.step, self.q)
            )

            self.distribution = self.get_uniform_distribution(low=self.low, high=self.high)
        elif self.type == "log_uniform":
            assert isinstance(self.low, float)
            assert isinstance(self.high, float)
            AbstractSearchableParameter._check_additional_params_passed(
                (self.choices, self.step, self.q)
            )

            self.distribution = self.get_log_uniform_distribution(
                low=self.low,
                high=self.high,
            )
        elif self.type == "discrete_uniform":
            assert isinstance(self.low, float)
            assert isinstance(self.high, float)
            assert isinstance(self.q, float)
            AbstractSearchableParameter._check_additional_params_passed((self.choices, self.step))

            self.distribution = self.get_discrete_uniform_distribution(
                low=self.low, high=self.high, q=self.q
            )

        elif self.type == "int_uniform":
            assert isinstance(self.low, int)
            assert isinstance(self.high, int)
            assert isinstance(self.step, int)
            AbstractSearchableParameter._check_additional_params_passed((self.choices, self.q))

            self.distribution = self.get_int_uniform_distribution(
                low=self.low, high=self.high, step=self.step
            )
        elif self.type == "log_int_uniform":
            assert isinstance(self.low, int)
            assert isinstance(self.high, int)
            assert isinstance(self.step, int)
            AbstractSearchableParameter._check_additional_params_passed((self.choices, self.q))
            self.distribution = self.get_log_int_uniform_distribution(
                low=self.low, high=self.high, step=self.step
            )
        else:
            raise ValueError(
                f"Parameter {self.type} is not supported. Pick one of {get_args(ParameterTypes)}"
            )

    @staticmethod
    @abc.abstractmethod
    def get_categorical_distribution(choices: Sequence[PrimitiveTypes]) -> Distribution:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_uniform_distribution(low: float, high: float) -> Distribution:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_log_uniform_distribution(low: float, high: float) -> Distribution:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_discrete_uniform_distribution(low: float, high: float, q: float) -> Distribution:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_int_uniform_distribution(low: int, high: int, step: int) -> Distribution:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_log_int_uniform_distribution(low: int, high: int, step: int) -> Distribution:
        pass


class SearchableParameter(AbstractSearchableParameter[optuna.distributions.BaseDistribution]):
    @staticmethod
    def get_categorical_distribution(
        choices: Sequence[PrimitiveTypes],
    ) -> optuna.distributions.BaseDistribution:
        return optuna.distributions.CategoricalDistribution(choices=choices)

    @staticmethod
    def get_uniform_distribution(low: float, high: float) -> optuna.distributions.BaseDistribution:
        return optuna.distributions.UniformDistribution(low=low, high=high)

    @staticmethod
    def get_log_uniform_distribution(
        low: float, high: float
    ) -> optuna.distributions.BaseDistribution:
        return optuna.distributions.LogUniformDistribution(low=low, high=high)

    @staticmethod
    def get_discrete_uniform_distribution(
        low: float, high: float, q: float
    ) -> optuna.distributions.BaseDistribution:
        return optuna.distributions.DiscreteUniformDistribution(low=low, high=high, q=q)

    @staticmethod
    def get_int_uniform_distribution(
        low: int, high: int, step: int
    ) -> optuna.distributions.BaseDistribution:
        return optuna.distributions.IntUniformDistribution(low=low, high=high, step=step)

    @staticmethod
    def get_log_int_uniform_distribution(
        low: int, high: int, step: int
    ) -> optuna.distributions.BaseDistribution:
        return optuna.distributions.IntLogUniformDistribution(low=low, high=high, step=step)