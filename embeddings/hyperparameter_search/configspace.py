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


class BaseConfigSpace(ABC):
    def _parse_parameter(
        self, param_name: str, trial: optuna.trial.Trial
    ) -> Tuple[str, PrimitiveTypes]:
        param: Parameter = self.__getattribute__(param_name)
        if isinstance(param, SearchableParameter):
            param.value = trial._suggest(name=param.name, distribution=param.distribution)
            return param.name, param.value
        elif isinstance(param, ConstantParameter):
            return param.name, param.value
        else:
            raise ValueError(
                f"Parameter type {type(param)} is not suported! "
                f"Supported types are: {SearchableParameter.__name__} and {ConstantParameter.__name__}"
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


# Mypy currently properly don't handle dataclasses with abstract methods  https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class AbstractFlairModelTrainerConfigSpace(BaseConfigSpace, ABC):
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


@dataclass
class FlairModelTrainerConfigSpace(AbstractFlairModelTrainerConfigSpace):
    embedding_type: Parameter = SearchableParameter(
        name="embedding_type",
        type="categorical",
        choices=["dynamic", "static"],
    )
    dynamic_embedding_name: Parameter = SearchableParameter(
        name="embedding_name",
        type="categorical",
        choices=["allegro/herbert-base-cased", "clarin-pl/herbert-kgr10"],
    )
    static_embedding_name: Parameter = SearchableParameter(
        name="embedding_name",
        type="categorical",
        choices=["clarin-pl/word2vec-kgr10", "clarin-pl/fastText-kgr10"],
    )

    def _map_embedding_parameters(
        self, trial: optuna.trial.Trial
    ) -> Tuple[Dict[str, PrimitiveTypes], Set[str]]:
        parameters = {}

        embedding_type_name, embedding_type_val = self._parse_parameter(
            trial=trial, param_name="embedding_type"
        )
        parameters[embedding_type_name] = embedding_type_val

        if embedding_type_val == "dynamic":
            parameters.update(
                [self._parse_parameter(trial=trial, param_name="dynamic_embedding_name")]
            )
        elif embedding_type_val == "static":
            parameters.update(
                [self._parse_parameter(trial=trial, param_name="static_embedding_name")]
            )
        else:
            raise ValueError("Embedding type not in ['static', 'dynamic']")

        mapped_parameters: Final = {
            "embedding_type",
            "dynamic_embedding_name",
            "static_embedding_name",
        }

        return parameters, mapped_parameters

    def sample_parameters(self, trial: optuna.trial.Trial) -> Dict[str, PrimitiveTypes]:
        embedding_params, mapped_embedding_params_names = self._map_embedding_parameters(trial)
        task_params, mapped_task_params_names = self._map_task_specific_parameters(trial)
        params = self._map_parameters(
            parameters_names=[
                param_name
                for param_name in self._get_annotations().keys()
                if param_name not in (mapped_task_params_names | mapped_embedding_params_names)
            ],
            trial=trial,
        )
        return {**params, **task_params, **embedding_params}

    @staticmethod
    def parse_parameters(parameters: Dict[str, PrimitiveTypes]) -> SampledParameters:
        embedding_type = parameters.pop("embedding_type")
        embedding_name = parameters.pop("embedding_name")
        assert isinstance(embedding_name, str)
        (
            parameters,
            task_train_kwargs,
        ) = FlairModelTrainerConfigSpace._parse_model_trainer_parameters(parameters)
        if len(parameters):
            raise ValueError(
                f"Some of the parameters are not mapped. Unmapped parameters: {parameters}"
            )
        return {
            "embedding_type": embedding_type,
            "embedding_name": embedding_name,
            "task_train_kwargs": task_train_kwargs,
        }


@dataclass
class SequenceLabelingConfigSpace(FlairModelTrainerConfigSpace):
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
        embedding_type = parameters.pop("embedding_type")
        assert isinstance(embedding_type, str)
        embedding_name = parameters.pop("embedding_name")
        assert isinstance(embedding_name, str)
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
            "embedding_type": embedding_type,
            "embedding_name": embedding_name,
            "hidden_size": hidden_size,
            "task_model_kwargs": task_model_kwargs,
            "task_train_kwargs": task_train_kwargs,
        }


@dataclass
class TextClassificationConfigSpace(FlairModelTrainerConfigSpace):
    dynamic_document_pooling: Parameter = SearchableParameter(
        name="document_pooling",
        type="categorical",
        choices=[
            "FlairDocumentCNNEmbeddings",
            "FlairDocumentRNNEmbeddings",
            "FlairTransformerDocumentEmbedding",
        ],
    )
    static_document_pooling: Parameter = SearchableParameter(
        name="document_pooling",
        type="categorical",
        choices=[
            "FlairDocumentCNNEmbeddings",
            "FlairDocumentRNNEmbeddings",
            "FlairDocumentPoolEmbedding",
        ],
    )
    static_pooling_strategy: Parameter = SearchableParameter(
        name="pooling_strategy", type="categorical", choices=["min", "max", "mean"]
    )
    dynamic_pooling_strategy: Parameter = SearchableParameter(
        name="pooling_strategy", type="categorical", choices=["cls", "max", "mean"]
    )
    static_fine_tune_mode: Parameter = SearchableParameter(
        name="fine_tune_mode", type="categorical", choices=["none", "linear", "nonlinear"]
    )
    dynamic_fine_tune: Parameter = SearchableParameter(
        name="fine_tune", type="categorical", choices=[True, False]
    )
    # TODO: choices to Optuna can only take primitives; typing error in mypy
    kernels: Parameter = SearchableParameter(
        name="kernels",
        type="categorical",
        choices=[((100, 3), (100, 4), (100, 5)), ((200, 4), (200, 5), (200, 6))],
    )
    hidden_size: Parameter = SearchableParameter(
        name="hidden_size", type="int_uniform", low=128, high=2048, step=128
    )
    rnn_type: Parameter = SearchableParameter(
        name="rnn_type", type="categorical", choices=["LSTM", "GRU"]
    )
    rnn_layers: Parameter = SearchableParameter(
        name="rnn_layers", type="int_uniform", low=1, high=3, step=1
    )
    bidirectional: Parameter = SearchableParameter(
        name="bidirectional", type="categorical", choices=[True, False]
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
    reproject_words: Parameter = SearchableParameter(
        name="reproject_words", type="categorical", choices=[True, False]
    )

    def _map_task_specific_parameters(
        self, trial: optuna.trial.Trial
    ) -> Tuple[Dict[str, PrimitiveTypes], Set[str]]:
        cnn_params = ("kernels",)
        rnn_params = ("hidden_size", "rnn_type", "rnn_layers", "bidirectional")
        shared_params = ("dropout", "word_dropout", "locked_dropout", "reproject_words")
        static_pooling_params = ("static_pooling_strategy", "static_fine_tune_mode")
        dynamic_pooling_params = ("dynamic_pooling_strategy", "dynamic_fine_tune")
        parameters = {}

        embedding_type_param: Parameter = self.__getattribute__("embedding_type")
        if embedding_type_param.value == "dynamic":
            document_pooling_name, document_pooling_val = self._parse_parameter(
                trial=trial, param_name="dynamic_document_pooling"
            )
        else:
            document_pooling_name, document_pooling_val = self._parse_parameter(
                trial=trial, param_name="static_document_pooling"
            )
        parameters[document_pooling_name] = document_pooling_val
        parameter_names: Tuple[str, ...]
        if document_pooling_val == "FlairDocumentCNNEmbeddings":
            parameter_names = cnn_params + shared_params
        elif document_pooling_val == "FlairDocumentRNNEmbeddings":
            parameter_names = rnn_params + shared_params
        elif document_pooling_val == "FlairTransformerDocumentEmbedding":
            parameter_names = dynamic_pooling_params
        elif document_pooling_val == "FlairDocumentPoolEmbedding":
            parameter_names = static_pooling_params
        else:
            raise ValueError(
                f"{document_pooling_val} not recognized as valid document pooling class."
            )

        parameters.update(self._map_parameters(parameters_names=list(parameter_names), trial=trial))

        mapped_parameters: Final[Set[str]] = set(self.__annotations__.keys())

        return parameters, mapped_parameters

    @staticmethod
    def parse_parameters(parameters: Dict[str, PrimitiveTypes]) -> SampledParameters:
        embedding_name = parameters.pop("embedding_name")
        assert isinstance(embedding_name, str)
        embedding_type = parameters.pop("embedding_type")
        assert isinstance(embedding_type, str)
        document_pooling = parameters.pop("document_pooling")
        assert isinstance(document_pooling, str)

        load_model_keys: Final = {
            "pooling_strategy",
            "fine_tune_mode",
            "fine_tune",
            "kernels",
            "hidden_size",
            "rnn_type",
            "rnn_layers",
            "bidirectional",
            "dropout",
            "word_dropout",
            "locked_dropout",
            "reproject_words",
        }
        load_model_kwargs = {k: parameters.pop(k) for k in load_model_keys if k in parameters}

        parameters, task_train_kwargs = SequenceLabelingConfigSpace._parse_model_trainer_parameters(
            parameters=parameters
        )

        if len(parameters):
            raise ValueError(
                f"Some of the parameters are not mapped. Unmapped parameters: {parameters}"
            )
        return {
            "embedding_name": embedding_name,
            "embedding_type": embedding_type,
            "document_pooling": document_pooling,
            "task_model_kwargs": {},
            "task_train_kwargs": task_train_kwargs,
            "load_model_kwargs": load_model_kwargs,
        }


ConfigSpace = TypeVar("ConfigSpace", bound=BaseConfigSpace)
