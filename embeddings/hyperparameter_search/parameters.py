import abc
from abc import ABC
from dataclasses import dataclass, field
from typing import Generic, Literal, Optional, Sequence, TypeVar, get_args

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


# Mypy currently properly don't handle dataclasses with abstract methods  https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class AbstractSearchableParameter(ABC, Generic[Distribution]):
    name: str
    type: ParameterTypes
    distribution: Distribution = field(init=False)
    low: Optional[Numeric] = None
    high: Optional[Numeric] = None
    step: Optional[Numeric] = None
    q: Optional[float] = None
    choices: Optional[Sequence[PrimitiveTypes]] = None

    def _check_arguments(self, forbidden_args: Sequence[str], required_args: Sequence[str]) -> None:
        for variable_name in forbidden_args:
            var = getattr(self, variable_name)
            if var:
                raise TypeError(
                    f'Argument "{variable_name}" cannot be set for SearchableParameter type: "{self.type}". '
                    f'Only {required_args} can be passed as argument for type: "{self.type}".'
                )

    def __post_init__(self) -> None:
        if self.type == "categorical":
            self._check_arguments(
                forbidden_args=("low", "high", "step", "q"), required_args=("choices")
            )
            assert isinstance(self.choices, Sequence)

            self.distribution = self.get_categorical_distribution(choices=self.choices)
        elif self.type == "uniform":
            self._check_arguments(
                forbidden_args=("choices", "step", "q"), required_args=("low", "high")
            )
            assert isinstance(self.low, float)
            assert isinstance(self.high, float)

            self.distribution = self.get_uniform_distribution(low=self.low, high=self.high)
        elif self.type == "log_uniform":
            self._check_arguments(
                forbidden_args=("choices", "step", "q"), required_args=("low", "high")
            )
            assert isinstance(self.low, float)
            assert isinstance(self.high, float)

            self.distribution = self.get_log_uniform_distribution(
                low=self.low,
                high=self.high,
            )
        elif self.type == "discrete_uniform":
            self._check_arguments(
                forbidden_args=("choices", "step"), required_args=("low", "high", "q")
            )
            assert isinstance(self.low, float)
            assert isinstance(self.high, float)
            assert isinstance(self.q, float)

            self.distribution = self.get_discrete_uniform_distribution(
                low=self.low, high=self.high, q=self.q
            )

        elif self.type == "int_uniform":
            self._check_arguments(
                forbidden_args=("choices", "q"), required_args=("low", "high", "step")
            )
            assert isinstance(self.low, int)
            assert isinstance(self.high, int)
            assert isinstance(self.step, int)

            self.distribution = self.get_int_uniform_distribution(
                low=self.low, high=self.high, step=self.step
            )
        elif self.type == "log_int_uniform":
            self._check_arguments(
                forbidden_args=("choices", "q"), required_args=("low", "high", "step")
            )
            assert isinstance(self.low, int)
            assert isinstance(self.high, int)
            assert isinstance(self.step, int)

            self.distribution = self.get_log_int_uniform_distribution(
                low=self.low, high=self.high, step=self.step
            )
        else:
            raise ValueError(
                f"ParameterSearch type: {self.type} is not supported. Pick one of {get_args(ParameterTypes)}"
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
