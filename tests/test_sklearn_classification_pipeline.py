from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import numpy as np
import pytest
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from embeddings.config.flair_config import FlairTextClassificationBasicConfig
from embeddings.data.data_loader import HuggingFaceDataLoader
from embeddings.data.dataset import Dataset
from embeddings.embedding.flair_loader import FlairDocumentPoolEmbeddingLoader
from embeddings.embedding.sklearn_embedding import SklearnEmbedding
from embeddings.embedding.vectorizer.flair import FlairDocumentVectorizer
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.sklearn_model import SklearnModel
from embeddings.pipeline.hf_preprocessing_pipeline import HuggingFacePreprocessingPipeline
from embeddings.pipeline.sklearn_classification import SklearnClassificationPipeline
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.sklearn_task.text_classification import TextClassification
from embeddings.transformation.hf_transformation.to_pandas_transformation import (
    ToPandasHuggingFaceCorpusTransformation,
)
from embeddings.transformation.pandas_transformation.rename_input_columns_transformation import (
    RenameInputColumnsTransformation,
)


@pytest.fixture(scope="module")
def dataset_name() -> str:
    return "clarin-pl/polemo2-official"


@pytest.fixture(scope="module")
def dataset_kwargs(dataset_name: str) -> Dict[str, Any]:
    return {
        "dataset_name_or_path": dataset_name,
        "input_column_name": "text",
        "target_column_name": "target",
    }


@pytest.fixture(scope="module")
def dataset_path(dataset_name: str) -> "TemporaryDirectory[str]":
    path = TemporaryDirectory()
    pipeline = HuggingFacePreprocessingPipeline(
        dataset_name=dataset_name,
        persist_path=path.name,
        sample_missing_splits=None,
        ignore_test_subset=False,
        seed=441,
    )
    pipeline.run()

    return path


@pytest.fixture(scope="module")
def local_dataset_kwargs(dataset_path: "TemporaryDirectory[str]") -> Dict[str, Any]:
    return {
        "dataset_name_or_path": dataset_path.name,
        "input_column_name": "text",
        "target_column_name": "target",
    }


@pytest.fixture(scope="module")
def embedding_kwargs() -> Dict[str, Any]:
    return {"max_features": 100}


@pytest.fixture(scope="module")
def flair_embedding_kwargs() -> Dict[str, Any]:
    return {"embedding_name": "clarin-pl/word2vec-kgr10", "model_type_reference": ""}


@pytest.fixture(scope="module")
def classifier_kwargs() -> Dict[str, Any]:
    return {"n_jobs": 4, "verbose": 1, "random_state": 441}


@pytest.fixture(scope="module")
def sklearn_classification_pipeline(
    dataset_kwargs: Dict[str, Any],
    embedding_kwargs: Dict[str, Any],
    result_path: "TemporaryDirectory[str]",
) -> Tuple[SklearnClassificationPipeline, "TemporaryDirectory[str]"]:
    return (
        SklearnClassificationPipeline(
            **dataset_kwargs,
            vectorizer_kwargs=embedding_kwargs,
            output_path=Path(result_path.name),
            classifier=AdaBoostClassifier
        ),
        result_path,
    )


@pytest.fixture(scope="module")
def sklearn_local_classification_pipeline(
    local_dataset_kwargs: Dict[str, Any],
    embedding_kwargs: Dict[str, Any],
    result_path: "TemporaryDirectory[str]",
) -> Tuple[SklearnClassificationPipeline, "TemporaryDirectory[str]"]:
    return (
        SklearnClassificationPipeline(
            **local_dataset_kwargs,
            vectorizer_kwargs=embedding_kwargs,
            output_path=Path(result_path.name),
            classifier=AdaBoostClassifier
        ),
        result_path,
    )


@pytest.fixture(scope="module")
def sklearn_flair_classification_pipeline(
    dataset_kwargs: Dict[str, Any],
    flair_embedding_kwargs: Dict[str, Any],
    classifier_kwargs: Dict[str, Any],
    result_path: "TemporaryDirectory[str]",
) -> Tuple[StandardPipeline, "TemporaryDirectory[str]"]:
    dataset = Dataset(dataset_kwargs["dataset_name_or_path"])
    data_loader = HuggingFaceDataLoader()
    transformation = ToPandasHuggingFaceCorpusTransformation().then(
        RenameInputColumnsTransformation(
            dataset_kwargs["input_column_name"], dataset_kwargs["target_column_name"]
        )
    )
    task = TextClassification(classifier=RandomForestClassifier, classifier_kwargs=classifier_kwargs)
    evaluator = TextClassificationEvaluator()
    config = FlairTextClassificationBasicConfig()
    flair_embedding_loader = FlairDocumentPoolEmbeddingLoader(**flair_embedding_kwargs)
    flair_embedding = flair_embedding_loader.get_embedding(
        config.document_embedding_cls, **config.load_model_kwargs
    )
    embedding = SklearnEmbedding(
        vectorizer_kwargs={"embedding": flair_embedding},
        vectorizer=FlairDocumentVectorizer,
    )
    model = SklearnModel(embedding, task, predict_subset="test")
    return (
        StandardPipeline(dataset, data_loader, transformation, model, evaluator),
        result_path,
    )


def test_sklearn_classification_pipeline(
    sklearn_classification_pipeline: Tuple[
        SklearnClassificationPipeline,
        "TemporaryDirectory[str]",
    ],
) -> None:
    pipeline, path = sklearn_classification_pipeline
    result = pipeline.run()
    np.testing.assert_almost_equal(result.accuracy, 0.62317, decimal=pytest.decimal)
    np.testing.assert_almost_equal(result.f1_macro, 0.60027, decimal=pytest.decimal)
    np.testing.assert_almost_equal(result.precision_macro, 0.59270, decimal=pytest.decimal)
    np.testing.assert_almost_equal(result.recall_macro, 0.61276, decimal=pytest.decimal)

    assert isinstance(result.data.y_pred, np.ndarray)
    assert isinstance(result.data.y_true, np.ndarray)
    assert result.data.y_pred.dtype == np.int64
    assert result.data.y_true.dtype == np.int64


def test_sklearn_local_classification_pipeline(
    sklearn_local_classification_pipeline: Tuple[
        SklearnClassificationPipeline,
        "TemporaryDirectory[str]",
    ],
) -> None:
    pipeline, path = sklearn_local_classification_pipeline
    result = pipeline.run()
    path.cleanup()
    np.testing.assert_almost_equal(result.accuracy, 0.62317, decimal=pytest.decimal)
    np.testing.assert_almost_equal(result.f1_macro, 0.60027, decimal=pytest.decimal)
    np.testing.assert_almost_equal(result.precision_macro, 0.59270, decimal=pytest.decimal)
    np.testing.assert_almost_equal(result.recall_macro, 0.61276, decimal=pytest.decimal)

    assert isinstance(result.data.y_pred, np.ndarray)
    assert isinstance(result.data.y_true, np.ndarray)
    assert result.data.y_pred.dtype == np.int64
    assert result.data.y_true.dtype == np.int64


def test_sklearn_flair_classification_pipeline(
    sklearn_flair_classification_pipeline: Tuple[
        StandardPipeline,
        "TemporaryDirectory[str]",
    ],
) -> None:
    pipeline, path = sklearn_flair_classification_pipeline
    result = pipeline.run()
    np.testing.assert_almost_equal(result.accuracy, 0.70365, decimal=pytest.decimal)
    np.testing.assert_almost_equal(result.f1_macro, 0.66031, decimal=pytest.decimal)
    np.testing.assert_almost_equal(result.precision_macro, 0.71139, decimal=pytest.decimal)
    np.testing.assert_almost_equal(result.recall_macro, 0.65517, decimal=pytest.decimal)

    assert isinstance(result.data.y_pred, np.ndarray)
    assert isinstance(result.data.y_true, np.ndarray)
    assert result.data.y_pred.dtype == np.int64
    assert result.data.y_true.dtype == np.int64
