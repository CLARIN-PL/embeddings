from abc import ABC
from dataclasses import InitVar, dataclass, field
from typing import Dict, List, Set, Tuple, Union

import optuna
from typing_extensions import Final

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


# Mypy currently properly don't handle dataclasses with abstract methods  https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class AbstractFlairModelTrainerConfigSpace(BaseConfigSpace, ABC):
    embedding_name: InitVar[Union[str, List[str]]]
    param_embedding_name: Parameter = field(init=False)
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
    def _parse_model_trainer_parameters(
        parameters: Dict[str, ParameterValues]
    ) -> Tuple[Dict[str, ParameterValues], Dict[str, ParameterValues]]:
        task_train_keys: Final = {
            "learning_rate",
            "mini_batch_size",
            "max_epochs",
            "param_selection_mode",
            "save_final_model",
        }
        task_train_kwargs = BaseConfigSpace._pop_parameters(
            parameters=parameters, parameters_keys=task_train_keys
        )
        return parameters, task_train_kwargs


class FlairModelTrainerConfigSpace(AbstractFlairModelTrainerConfigSpace):
    @classmethod
    def parse_parameters(cls, parameters: Dict[str, ParameterValues]) -> SampledParameters:
        embedding_name = parameters.pop("embedding_name")
        (
            parameters,
            task_train_kwargs,
        ) = FlairModelTrainerConfigSpace._parse_model_trainer_parameters(parameters)
        BaseConfigSpace._check_unmapped_parameters(parameters=parameters)
        return {"embedding_name": embedding_name, "task_train_kwargs": task_train_kwargs}


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
    ) -> Tuple[Dict[str, ParameterValues], Set[str]]:
        parameters = {}
        use_rnn_name, use_rnn_val = self._parse_parameter(trial=trial, param_name="use_rnn")
        parameters[use_rnn_name] = use_rnn_val

        if use_rnn_val:
            for rnn_param in ("rnn_layers", "rnn_type"):
                parameters.update([self._parse_parameter(trial=trial, param_name=rnn_param)])

        mapped_parameters: Final[Set[str]] = {"rnn_layers", "rnn_type", "use_rnn"}
        return parameters, mapped_parameters

    @classmethod
    def parse_parameters(cls, parameters: Dict[str, ParameterValues]) -> SampledParameters:
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
        task_model_kwargs = BaseConfigSpace._pop_parameters(
            parameters=parameters, parameters_keys=task_model_keys
        )
        parameters, task_train_kwargs = SequenceLabelingConfigSpace._parse_model_trainer_parameters(
            parameters=parameters
        )
        BaseConfigSpace._check_unmapped_parameters(parameters=parameters)

        return {
            "embedding_name": embedding_name,
            "hidden_size": hidden_size,
            "task_model_kwargs": task_model_kwargs,
            "task_train_kwargs": task_train_kwargs,
        }


@dataclass
class TextClassificationConfigSpace(AbstractFlairModelTrainerConfigSpace):
    dynamic_document_embedding: Parameter = SearchableParameter(
        name="document_embedding",
        type="categorical",
        choices=[
            "FlairDocumentCNNEmbeddings",
            "FlairDocumentRNNEmbeddings",
            "FlairTransformerDocumentEmbedding",
        ],
    )
    static_document_embedding: Parameter = SearchableParameter(
        name="document_embedding",
        type="categorical",
        choices=[
            "FlairDocumentCNNEmbeddings",
            "FlairDocumentRNNEmbeddings",
            "FlairDocumentPoolEmbedding",
        ],
    )
    static_pooling: Parameter = SearchableParameter(
        name="pooling", type="categorical", choices=["min", "max", "mean"]
    )
    dynamic_pooling: Parameter = SearchableParameter(
        name="pooling", type="categorical", choices=["cls", "max", "mean"]
    )
    static_fine_tune_mode: Parameter = SearchableParameter(
        name="fine_tune_mode", type="categorical", choices=["none", "linear", "nonlinear"]
    )
    dynamic_fine_tune: Parameter = SearchableParameter(
        name="fine_tune", type="categorical", choices=[False, True]
    )
    # Choices to Optuna can only take primitives;
    # This parameter results in Optuna warning but the library works properly
    cnn_pool_kernels: Parameter = SearchableParameter(
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
    reproject_words: Parameter = SearchableParameter(
        name="reproject_words", type="categorical", choices=[True, False]
    )

    def get_embedding_type(self) -> str:
        embedding_name_param: Parameter = self.__getattribute__("param_embedding_name")
        embedding_name = embedding_name_param.value
        assert isinstance(embedding_name, str)
        embedding_type = self._retrieve_embedding_type(embedding_name=embedding_name)
        assert isinstance(embedding_type, str)
        return embedding_type

    def _map_task_specific_parameters(
        self, trial: optuna.trial.Trial
    ) -> Tuple[Dict[str, ParameterValues], Set[str]]:
        shared_params = ("dropout", "word_dropout", "reproject_words")
        param_names_mapping: Final = {
            "FlairDocumentCNNEmbeddings": ("cnn_pool_kernels",) + shared_params,
            "FlairDocumentRNNEmbeddings": ("hidden_size", "rnn_type", "rnn_layers", "bidirectional")
            + shared_params,
            "FlairTransformerDocumentEmbedding": ("dynamic_pooling", "dynamic_fine_tune"),
            "FlairDocumentPoolEmbedding": ("static_pooling", "static_fine_tune_mode"),
        }
        parameters = {}

        embedding_name, embedding_name_val = self._parse_parameter(
            trial=trial, param_name="param_embedding_name"
        )
        parameters[embedding_name] = embedding_name_val

        embedding_type_param: str = self.get_embedding_type()
        assert embedding_type_param in ["dynamic", "static"]

        document_embedding_name, document_embedding_val = self._parse_parameter(
            trial=trial, param_name=f"{embedding_type_param}_document_embedding"
        )
        if not isinstance(document_embedding_val, str):
            raise TypeError("Variable document_embedding_val must be a str!")

        parameters[document_embedding_name] = document_embedding_val
        parameter_names = param_names_mapping[document_embedding_val]

        parameters.update(self._map_parameters(parameters_names=list(parameter_names), trial=trial))

        mapped_parameters: Final[Set[str]] = {
            "param_embedding_name",
            *list(self.__annotations__.keys()),
        }

        return parameters, mapped_parameters

    @classmethod
    def parse_parameters(cls, parameters: Dict[str, ParameterValues]) -> SampledParameters:
        embedding_name = parameters.pop("embedding_name")
        assert isinstance(embedding_name, str)
        document_embedding = parameters.pop("document_embedding")
        assert isinstance(document_embedding, str)

        load_model_keys: Final = {
            "pooling",
            "fine_tune_mode",
            "fine_tune",
            "kernels",
            "hidden_size",
            "rnn_type",
            "rnn_layers",
            "bidirectional",
            "dropout",
            "word_dropout",
            "reproject_words",
        }
        load_model_kwargs = BaseConfigSpace._pop_parameters(
            parameters=parameters, parameters_keys=load_model_keys
        )

        parameters, task_train_kwargs = SequenceLabelingConfigSpace._parse_model_trainer_parameters(
            parameters=parameters
        )
        BaseConfigSpace._check_unmapped_parameters(parameters=parameters)

        return {
            "embedding_name": embedding_name,
            "document_embedding": document_embedding,
            "task_model_kwargs": {},
            "task_train_kwargs": task_train_kwargs,
            "load_model_kwargs": load_model_kwargs,
        }
