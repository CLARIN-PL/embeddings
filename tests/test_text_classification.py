from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import datasets
import flair
import numpy as np
import pytest
from flair.data import Corpus
from numpy import typing as nptyping

from embeddings.data.data_loader import HuggingFaceDataLoader
from embeddings.data.dataset import Dataset
from embeddings.embedding.auto_flair import AutoFlairDocumentEmbedding
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.flair_task.text_classification import TextClassification
from embeddings.transformation.flair_transformation.classification_corpus_transformation import (
    ClassificationCorpusTransformation,
)
from embeddings.transformation.flair_transformation.downsample_corpus_transformation import (
    DownsampleFlairCorpusTransformation,
)


@pytest.fixture(scope="module")
def text_classification_pipeline(
    result_path: "TemporaryDirectory[str]",
) -> Tuple[
    StandardPipeline[
        str, datasets.DatasetDict, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]
    ],
    "TemporaryDirectory[str]",
]:
    dataset = Dataset(
        "clarin-pl/polemo2-official",
        train_domains=["reviews"],
        dev_domains=["reviews"],
        test_domains=["reviews"],
        text_cfg="sentence",
    )
    data_loader = HuggingFaceDataLoader()
    transformation = ClassificationCorpusTransformation("text", "target").then(
        DownsampleFlairCorpusTransformation(*(0.01, 0.01, 0.01), stratify=False)
    )
    embedding = AutoFlairDocumentEmbedding.from_hub("allegro/herbert-base-cased")
    task = TextClassification(result_path.name, task_train_kwargs={"max_epochs": 1})
    model = FlairModel(embedding, task)
    evaluator = TextClassificationEvaluator()
    pipeline = StandardPipeline(dataset, data_loader, transformation, model, evaluator)
    return pipeline, result_path


def test_text_classification_pipeline(
    text_classification_pipeline: Tuple[
        StandardPipeline[
            str, datasets.DatasetDict, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]
        ],
        "TemporaryDirectory[str]",
    ],
) -> None:
    flair.set_seed(441)
    pipeline, path = text_classification_pipeline
    result = pipeline.run()
    np.testing.assert_almost_equal(result["accuracy"]["accuracy"], 0.3333333)
    np.testing.assert_almost_equal(result["f1__average_macro"]["f1"], 0.1666666)
    np.testing.assert_almost_equal(result["precision__average_macro"]["precision"], 0.1111111)
    np.testing.assert_almost_equal(result["recall__average_macro"]["recall"], 0.3333333)

    task_from_ckpt = TextClassification.from_checkpoint(
        checkpoint_path=(Path(path.name) / "final-model.pt"), output_path=path.name
    )
    loaded_data = pipeline.data_loader.load(pipeline.dataset)
    transformed_data = pipeline.transformation.transform(loaded_data)
    test_data = transformed_data.test

    y_pred, loss = task_from_ckpt.predict(test_data)
    y_true = task_from_ckpt.get_y(test_data, task_from_ckpt.y_type, task_from_ckpt.y_dictionary)
    results_from_ckpt = pipeline.evaluator.evaluate({"y_pred": y_pred, "y_true": y_true})
    assert np.array_equal(result["data"]["y_pred"], results_from_ckpt["data"]["y_pred"])

    path.cleanup()


@pytest.fixture(scope="module")
def text_classification_pipeline_local_embedding(
    local_embedding_filepath: Path,
    result_path: "TemporaryDirectory[str]",
    model_type_reference: str = "embeddings.embedding.static.word2vec.IPIPANWord2VecEmbedding",
) -> Tuple[
    StandardPipeline[
        str, datasets.DatasetDict, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]
    ],
    "TemporaryDirectory[str]",
]:
    dataset = Dataset(
        "clarin-pl/polemo2-official",
        train_domains=["reviews"],
        dev_domains=["reviews"],
        test_domains=["reviews"],
        text_cfg="sentence",
    )
    data_loader = HuggingFaceDataLoader()
    transformation = ClassificationCorpusTransformation("text", "target").then(
        DownsampleFlairCorpusTransformation(*(0.01, 0.01, 0.01), stratify=False)
    )
    embedding = AutoFlairDocumentEmbedding.from_file(local_embedding_filepath, model_type_reference)
    task = TextClassification(result_path.name, task_train_kwargs={"max_epochs": 1})
    model = FlairModel(embedding, task)
    evaluator = TextClassificationEvaluator()
    pipeline = StandardPipeline(dataset, data_loader, transformation, model, evaluator)
    return pipeline, result_path


def test_text_classification_pipeline_local_embedding(
    text_classification_pipeline_local_embedding: Tuple[
        StandardPipeline[
            str, datasets.DatasetDict, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]
        ],
        "TemporaryDirectory[str]",
    ],
) -> None:
    flair.set_seed(441)
    pipeline, path = text_classification_pipeline_local_embedding
    result = pipeline.run()

    np.testing.assert_almost_equal(result["accuracy"]["accuracy"], 0.3333333)
    np.testing.assert_almost_equal(result["f1__average_macro"]["f1"], 0.3333333)
    np.testing.assert_almost_equal(result["precision__average_macro"]["precision"], 0.3333333)
    np.testing.assert_almost_equal(result["recall__average_macro"]["recall"], 0.3333333)

    task_from_ckpt = TextClassification.from_checkpoint(
        checkpoint_path=(Path(path.name) / "final-model.pt"), output_path=path.name
    )
    loaded_data = pipeline.data_loader.load(pipeline.dataset)
    transformed_data = pipeline.transformation.transform(loaded_data)
    test_data = transformed_data.test

    y_pred, loss = task_from_ckpt.predict(test_data)
    y_true = task_from_ckpt.get_y(test_data, task_from_ckpt.y_type, task_from_ckpt.y_dictionary)
    results_from_ckpt = pipeline.evaluator.evaluate({"y_pred": y_pred, "y_true": y_true})
    assert np.array_equal(result["data"]["y_pred"], results_from_ckpt["data"]["y_pred"])

    path.cleanup()
