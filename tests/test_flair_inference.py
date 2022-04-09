from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict

import datasets
import flair
import numpy as np
import pytest
from _pytest.tmpdir import TempdirFactory
from flair.data import Corpus
from numpy import typing as nptyping

from embeddings.data.data_loader import HuggingFaceDataLoader
from embeddings.data.dataset import Dataset
from embeddings.embedding.auto_flair import AutoFlairDocumentEmbedding
from embeddings.embedding.flair_loader import FlairWordEmbeddingLoader
from embeddings.evaluator.sequence_labeling_evaluator import SequenceLabelingEvaluator
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.flair_task.sequence_labeling import SequenceLabeling
from embeddings.task.flair_task.text_classification import TextClassification
from embeddings.transformation.flair_transformation.classification_corpus_transformation import (
    ClassificationCorpusTransformation,
)
from embeddings.transformation.flair_transformation.column_corpus_transformation import (
    ColumnCorpusTransformation,
)
from embeddings.transformation.flair_transformation.downsample_corpus_transformation import (
    DownsampleFlairCorpusTransformation,
)
from embeddings.transformation.flair_transformation.split_sample_corpus_transformation import (
    SampleSplitsFlairCorpusTransformation,
)


@pytest.fixture(scope="module")
def tmp_path_module(tmpdir_factory: TempdirFactory) -> Path:
    path = tmpdir_factory.mktemp(__name__)
    return Path(path)


@pytest.fixture(scope="module")
def text_classification_pipeline(
    tmp_path_module: "TemporaryDirectory[str]",
) -> StandardPipeline[
    str, datasets.DatasetDict, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]
]:
    output_path = tmp_path_module
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
    task = TextClassification(output_path.name, task_train_kwargs={"max_epochs": 1})
    model = FlairModel(embedding, task)
    evaluator = TextClassificationEvaluator()
    pipeline = StandardPipeline(dataset, data_loader, transformation, model, evaluator)
    return pipeline


@pytest.fixture(scope="module")
def sequence_labeling_pipeline(
    tmp_path_module: "TemporaryDirectory[str]",
) -> StandardPipeline[
    str, datasets.DatasetDict, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]
]:
    output_path = tmp_path_module
    dataset = Dataset("clarin-pl/kpwr-ner")
    data_loader = HuggingFaceDataLoader()
    transformation = (
        ColumnCorpusTransformation("tokens", "ner")
        .then(SampleSplitsFlairCorpusTransformation(dev_fraction=0.1, seed=441))
        .then(DownsampleFlairCorpusTransformation(*(0.005, 0.01, 0.01), stratify=False))
    )
    task = SequenceLabeling(
        output_path.name,
        hidden_size=256,
        task_train_kwargs={"max_epochs": 1, "mini_batch_size": 64},
    )
    embedding_loader = FlairWordEmbeddingLoader("allegro/herbert-base-cased", "model_type_reference")
    embedding = embedding_loader.get_embedding()
    model = FlairModel(embedding, task)
    evaluator = SequenceLabelingEvaluator()
    pipeline = StandardPipeline(dataset, data_loader, transformation, model, evaluator)
    return pipeline


def test_text_classification_inference(
    text_classification_pipeline: StandardPipeline[
        str, datasets.DatasetDict, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]
    ],
    tmp_path_module: "TemporaryDirectory[str]",
) -> None:
    flair.set_seed(441)
    output_path = tmp_path_module
    pipeline = text_classification_pipeline
    result = pipeline.run()

    task_from_ckpt = TextClassification.from_checkpoint(
        checkpoint_path=(Path(output_path.name) / "final-model.pt"), output_path=output_path.name
    )
    loaded_data = pipeline.data_loader.load(pipeline.dataset)
    transformed_data = pipeline.transformation.transform(loaded_data)
    test_data = transformed_data.test

    y_pred, loss = task_from_ckpt.predict(test_data)
    y_true = task_from_ckpt.get_y(test_data, task_from_ckpt.y_type, task_from_ckpt.y_dictionary)
    results_from_ckpt = pipeline.evaluator.evaluate({"y_pred": y_pred, "y_true": y_true})
    assert np.array_equal(result["data"]["y_pred"], results_from_ckpt["data"]["y_pred"])


def test_sequence_labeling_inference(
    sequence_labeling_pipeline: StandardPipeline[
        str, datasets.DatasetDict, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]
    ],
    tmp_path_module: "TemporaryDirectory[str]",
) -> None:
    flair.set_seed(441)
    output_path = tmp_path_module
    pipeline = sequence_labeling_pipeline
    result = pipeline.run()

    task_from_ckpt = SequenceLabeling.from_checkpoint(
        checkpoint_path=(Path(output_path.name) / "final-model.pt"), output_path=output_path.name
    )
    loaded_data = pipeline.data_loader.load(pipeline.dataset)
    transformed_data = pipeline.transformation.transform(loaded_data)
    test_data = transformed_data.test

    y_pred, loss = task_from_ckpt.predict(test_data)
    y_true = task_from_ckpt.get_y(test_data, task_from_ckpt.y_type, task_from_ckpt.y_dictionary)
    results_from_ckpt = pipeline.evaluator.evaluate({"y_pred": y_pred, "y_true": y_true})

    assert np.array_equal(result["data"]["y_pred"], results_from_ckpt["data"]["y_pred"])
