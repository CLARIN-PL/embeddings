from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple, Union

import optuna
import pytest
from pydantic import create_model_from_typeddict

from embeddings.config.flair_config import (
    FlairSequenceLabelingAdvancedConfig,
    FlairTextClassificationAdvancedConfig,
)
from embeddings.config.lightning_config import LightningAdvancedConfig, LightningBasicConfig
from embeddings.config.parameters import SearchableParameter
from embeddings.data.dataset import LightingDataModuleSubset
from embeddings.pipeline.flair_classification import FlairClassificationPipeline
from embeddings.pipeline.flair_pair_classification import FlairPairClassificationPipeline
from embeddings.pipeline.flair_sequence_labeling import FlairSequenceLabelingPipeline
from embeddings.pipeline.pipelines_metadata import (
    FlairClassificationPipelineMetadata,
    FlairPairClassificationPipelineMetadata,
    FlairSequenceLabelingPipelineMetadata,
    LightningPipelineMetadata,
)
from embeddings.utils.utils import PrimitiveTypes


@pytest.fixture(scope="module")
def output_path() -> "TemporaryDirectory[str]":
    return TemporaryDirectory()


@pytest.fixture(scope="module")
def flair_text_classification_dataset_kwargs() -> Dict[str, PrimitiveTypes]:
    return {
        "dataset_name": "clarin-pl/polemo2-official",
        "input_column_name": "text",
        "target_column_name": "target",
        "load_dataset_kwargs": None,
    }


@pytest.fixture(scope="module")
def flair_text_pair_classification_dataset_kwargs() -> Dict[
    str, Union[PrimitiveTypes, Tuple[str, str]]
]:
    return {
        "dataset_name": "clarin-pl/cst-wikinews",
        "input_columns_names_pair": ("sentence_1", "sentence_2"),
        "target_column_name": "label",
        "load_dataset_kwargs": None,
    }


@pytest.fixture(scope="module")
def flair_sequence_labeling_dataset_kwargs() -> Dict[str, PrimitiveTypes]:
    return {
        "dataset_name": "clarin-pl/kpwr-ner",
        "input_column_name": "tokens",
        "target_column_name": "ner",
        "load_dataset_kwargs": None,
    }


@pytest.fixture(scope="module")
def flair_pipeline_kwargs(output_path: "TemporaryDirectory[str]") -> Dict[str, PrimitiveTypes]:
    return {"output_path": output_path.name, "embedding_name": "clarin-pl/roberta-polish-kgr10"}


@pytest.fixture(scope="module")
def flair_sequence_labeling_pipeline_kwargs() -> Dict[str, PrimitiveTypes]:
    return {"evaluation_mode": "conll", "tagging_scheme": None}


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
def flair_classification_pipeline_metadata(
    flair_pipeline_kwargs,
    flair_text_classification_dataset_kwargs,
) -> Dict[str, Any]:
    return {
        **flair_pipeline_kwargs,
        **flair_text_classification_dataset_kwargs,
        **{
            "config": FlairTextClassificationAdvancedConfig(
                document_embedding_cls="FlairDocumentPoolEmbedding",
                load_model_kwargs={},
                task_model_kwargs={},
                task_train_kwargs={},
            )
        },
    }


@pytest.fixture(scope="module")
def flair_pair_classification_pipeline_metadata(
    flair_pipeline_kwargs,
    flair_text_pair_classification_dataset_kwargs,
) -> Dict[str, Any]:
    return {
        **flair_pipeline_kwargs,
        **flair_text_pair_classification_dataset_kwargs,
        **{
            "config": FlairTextClassificationAdvancedConfig(
                document_embedding_cls="FlairDocumentPoolEmbedding",
                load_model_kwargs={},
                task_model_kwargs={},
                task_train_kwargs={},
            )
        },
    }


@pytest.fixture(scope="module")
def flair_sequence_labeling_pipeline_metadata(
    flair_pipeline_kwargs,
    flair_sequence_labeling_pipeline_kwargs,
    flair_sequence_labeling_dataset_kwargs,
) -> Dict[str, Any]:
    return {
        **flair_pipeline_kwargs,
        **flair_sequence_labeling_pipeline_kwargs,
        **flair_sequence_labeling_dataset_kwargs,
        **{
            "config": FlairSequenceLabelingAdvancedConfig(
                hidden_size=16, load_model_kwargs={}, task_model_kwargs={}, task_train_kwargs={}
            )
        },
    }


@pytest.fixture(scope="module")
def lightning_classification_pipeline_metadata(
    lightning_text_classification_dataset_kwargs, lightning_classification_kwargs
) -> Dict[str, Any]:
    return {**lightning_text_classification_dataset_kwargs, **lightning_classification_kwargs}


def test_flair_classification_pipeline_metadata(flair_classification_pipeline_metadata) -> None:
    metadata = FlairClassificationPipelineMetadata(**flair_classification_pipeline_metadata)
    FlairClassificationPipeline(**metadata)


def test_flair_pair_classification_pipeline_metadata(
    flair_pair_classification_pipeline_metadata,
) -> None:
    metadata = FlairPairClassificationPipelineMetadata(
        **flair_pair_classification_pipeline_metadata
    )
    FlairPairClassificationPipeline(**metadata)


def test_flair_sequence_labeling_pipeline_metadata(
    flair_sequence_labeling_pipeline_metadata,
) -> None:
    metadata = FlairSequenceLabelingPipelineMetadata(**flair_sequence_labeling_pipeline_metadata)
    FlairSequenceLabelingPipeline(**metadata)


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
