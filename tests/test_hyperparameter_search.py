from tempfile import TemporaryDirectory
from typing import Any, Dict

import optuna
import pytest

from embeddings.config.lightning_config import LightningBasicConfig
from embeddings.config.parameters import SearchableParameter
from embeddings.data.dataset import LightingDataModuleSubset
from embeddings.pipeline.pipelines_metadata import LightningPipelineMetadata
from embeddings.utils.utils import PrimitiveTypes


@pytest.fixture(scope="module")
def output_path() -> "TemporaryDirectory[str]":
    return TemporaryDirectory()


@pytest.fixture(scope="module")
def lightning_text_classification_dataset_kwargs() -> Dict[str, PrimitiveTypes]:
    return {
        "dataset_name_or_path": "clarin-pl/polemo2-official",
        "input_column_name": "text",
        "target_column_name": "target",
        "load_dataset_kwargs": None,
    }


@pytest.fixture(scope="module")
def lightning_classification_kwargs(output_path: "TemporaryDirectory[str]") -> Dict[str, Any]:
    return {
        "output_path": output_path.name,
        "embedding_name_or_path": "clarin-pl/roberta-polish-kgr10",
        "config": LightningBasicConfig(batch_size=1, finetune_last_n_layers=0),
        "tokenizer_name_or_path": None,
        "predict_subset": LightingDataModuleSubset.TEST,
    }


@pytest.fixture(scope="module")
def lightning_classification_pipeline_metadata(
    lightning_text_classification_dataset_kwargs: Dict[str, PrimitiveTypes],
    lightning_classification_kwargs: Dict[str, PrimitiveTypes],
) -> Dict[str, Any]:
    return {**lightning_text_classification_dataset_kwargs, **lightning_classification_kwargs}


def test_lightning_classification_pipeline_metadata(
    lightning_classification_pipeline_metadata,
) -> None:
    metadata = LightningPipelineMetadata(**lightning_classification_pipeline_metadata)
    LightningPipelineMetadata(**metadata)


def test_categorical_parameter() -> None:
    parameter = SearchableParameter(name="test", type="categorical", choices=[0, 1])
    assert isinstance(parameter.distribution, optuna.distributions.CategoricalDistribution)
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="categorical", choices=0)  # type: ignore
    with pytest.raises(TypeError):
        SearchableParameter(name="test", type="categorical", choices=[0, 1], q=0)
    with pytest.raises(TypeError):
        SearchableParameter(name="test", type="categorical", choices=[0, 1], low=0)
    with pytest.raises(TypeError):
        SearchableParameter(name="test", type="categorical", choices=[0, 1], high=100)
    with pytest.raises(TypeError):
        SearchableParameter(name="test", type="categorical", choices=[0, 5, 10], step=5)


def test_uniform_parameter() -> None:
    parameter = SearchableParameter(name="test", type="uniform", low=0.0, high=1.0)
    assert isinstance(parameter.distribution, optuna.distributions.UniformDistribution)
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="uniform", low=[0, 1], high=[0, 2])  # type: ignore
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="uniform", low=0.0)
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="uniform", high=1.0)
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="uniform", low=0, high=1)
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="uniform", low=0.0, high=1)
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="uniform", low=0, high=1.0)
    with pytest.raises(TypeError):
        SearchableParameter(name="test", type="uniform", low=0.0, high=1.0, q=0)
    with pytest.raises(TypeError):
        SearchableParameter(name="test", type="uniform", low=0.0, high=1.0, choices=[0, 1])
    with pytest.raises(TypeError):
        SearchableParameter(name="test", type="uniform", low=0.0, high=1.0, step=0.5)


def test_log_uniform_parameter() -> None:
    parameter = SearchableParameter(name="test", type="log_uniform", low=0.01, high=1.0)
    assert isinstance(parameter.distribution, optuna.distributions.LogUniformDistribution)
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="log_uniform", low=[0, 1], high=[0, 2])  # type: ignore
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="log_uniform", low=0.0)
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="log_uniform", high=1.0)
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="log_uniform", low=0, high=1)
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="log_uniform", low=0.0, high=1)
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="log_uniform", low=0, high=1.0)
    with pytest.raises(ValueError):
        SearchableParameter(name="test", type="log_uniform", low=0.0, high=1.0)
    with pytest.raises(TypeError):
        SearchableParameter(name="test", type="log_uniform", low=0.0, high=1.0, q=0)
    with pytest.raises(TypeError):
        SearchableParameter(name="test", type="log_uniform", low=0.0, high=1.0, choices=[0, 1])
    with pytest.raises(TypeError):
        SearchableParameter(name="test", type="log_uniform", low=0.0, high=1.0, step=0.5)


def test_discrete_uniform_parameter() -> None:
    parameter = SearchableParameter(
        name="test", type="discrete_uniform", low=0.01, high=1.0, q=0.01
    )
    assert isinstance(parameter.distribution, optuna.distributions.DiscreteUniformDistribution)
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="discrete_uniform", low=[0, 1], high=[0, 2], q=0)  # type: ignore
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="discrete_uniform", low=0.0)
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="discrete_uniform", high=1.0)
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="discrete_uniform", low=0, high=1)
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="discrete_uniform", low=0.0, high=1)
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="discrete_uniform", low=0, high=1.0)
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="discrete_uniform", low=0.0, high=1.0)
    with pytest.raises(AssertionError):
        SearchableParameter(name="test", type="discrete_uniform", low=0.0, high=1.0, q=0)
    with pytest.raises(TypeError):
        SearchableParameter(name="test", type="discrete_uniform", low=0.0, high=1.0, choices=[0, 1])
    with pytest.raises(TypeError):
        SearchableParameter(name="test", type="discrete_uniform", low=0.0, high=1.0, step=0.5)
