from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple, Union

import optuna
import pytest
from pydantic import create_model_from_typeddict

from embeddings.hyperparameter_search.parameters import SearchableParameter
from embeddings.pipeline.hugging_face_classification import HuggingFaceClassificationPipeline
from embeddings.pipeline.hugging_face_pair_classification import (
    HuggingFacePairClassificationPipeline,
)
from embeddings.pipeline.hugging_face_sequence_labeling import HuggingFaceSequenceLabelingPipeline
from embeddings.pipeline.pipelines_metadata import (
    HuggingFaceClassificationPipelineMetadata,
    HuggingFacePairClassificationPipelineMetadata,
    HuggingFaceSequenceLabelingPipelineMetadata,
)
from embeddings.utils.utils import PrimitiveTypes


@pytest.fixture
def output_path() -> TemporaryDirectory[str]:
    return TemporaryDirectory()


@pytest.fixture
def text_classification_dataset_kwargs() -> Dict[str, PrimitiveTypes]:
    return {
        "dataset_name": "clarin-pl/polemo2-official",
        "input_column_name": "text",
        "target_column_name": "target",
        "load_dataset_kwargs": None,
    }


@pytest.fixture
def text_pair_classification_dataset_kwargs() -> Dict[str, Union[PrimitiveTypes, Tuple[str, str]]]:
    return {
        "dataset_name": "clarin-pl/cst-wikinews",
        "input_columns_names_pair": ("sentence_1", "sentence_2"),
        "target_column_name": "label",
        "load_dataset_kwargs": None,
    }


@pytest.fixture
def sequence_labeling_dataset_kwargs() -> Dict[str, PrimitiveTypes]:
    return {
        "dataset_name": "clarin-pl/kpwr-ner",
        "input_column_name": "tokens",
        "target_column_name": "ner",
        "load_dataset_kwargs": None,
    }


@pytest.fixture
def pipeline_kwargs(output_path: TemporaryDirectory[str]) -> Dict[str, PrimitiveTypes]:
    return {
        "output_path": output_path.name,
        "embedding_name": "clarin-pl/roberta-polish-kgr10",
        "document_embedding_cls": "FlairDocumentPoolEmbedding",
        "load_model_kwargs": None,
        "task_model_kwargs": None,
        "task_train_kwargs": None,
    }


@pytest.fixture
def sequence_labeling_pipeline_kwargs() -> Dict[str, PrimitiveTypes]:
    return {"evaluation_mode": "conll", "tagging_scheme": None, "hidden_size": 128}


@pytest.fixture
def hf_classification_pipeline_metadata(
    pipeline_kwargs: Dict[str, str],
    text_classification_dataset_kwargs: Dict[str, PrimitiveTypes],
) -> Dict[str, Any]:
    return {
        **pipeline_kwargs,
        **text_classification_dataset_kwargs,
    }


@pytest.fixture
def hf_pair_classification_pipeline_metadata(
    pipeline_kwargs: Dict[str, str],
    text_pair_classification_dataset_kwargs: Dict[str, PrimitiveTypes],
) -> Dict[str, Any]:
    return {
        **pipeline_kwargs,
        **text_pair_classification_dataset_kwargs,
    }


@pytest.fixture
def hf_sequence_labeling_pipeline_metadata(
    pipeline_kwargs: Dict[str, str],
    sequence_labeling_pipeline_kwargs: Dict[str, PrimitiveTypes],
    sequence_labeling_dataset_kwargs: Dict[str, PrimitiveTypes],
) -> Dict[str, Any]:
    return {
        **pipeline_kwargs,
        **sequence_labeling_pipeline_kwargs,
        **sequence_labeling_dataset_kwargs,
    }


# Pydantic create_model_from_typeddict in 1.8.2 is no compilant with mypy
# https://github.com/samuelcolvin/pydantic/issues/3008
# It should be fixed in further release of pydantic library
def test_hf_classification_pipeline_metadata(
    hf_classification_pipeline_metadata: Dict[str, Any]
) -> None:
    metadata = create_model_from_typeddict(HuggingFaceClassificationPipelineMetadata)(  # type: ignore
        **hf_classification_pipeline_metadata
    ).dict()
    HuggingFaceClassificationPipeline(**metadata)


def test_hf_pair_classification_pipeline_metadata(
    hf_pair_classification_pipeline_metadata: Dict[str, Any]
) -> None:
    metadata = create_model_from_typeddict(HuggingFacePairClassificationPipelineMetadata)(  # type: ignore
        **hf_pair_classification_pipeline_metadata
    ).dict()
    HuggingFacePairClassificationPipeline(**metadata)


def test_hf_sequence_labeling_pipeline_metadata(
    hf_sequence_labeling_pipeline_metadata: Dict[str, Any]
) -> None:
    metadata = create_model_from_typeddict(HuggingFaceSequenceLabelingPipelineMetadata)(  # type: ignore
        **hf_sequence_labeling_pipeline_metadata
    ).dict()
    HuggingFaceSequenceLabelingPipeline(**metadata)


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
