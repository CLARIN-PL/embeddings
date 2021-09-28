from tempfile import TemporaryDirectory

import datasets
import flair
import pytest
from flair.data import Corpus

from embeddings.data.data_loader import ConllFlairCorpusDataLoader, HuggingFaceDataLoader
from embeddings.data.dataset import HuggingFaceDataset, LocalDataset
from embeddings.transformation.flair_transformation.column_corpus_transformation import (
    ColumnCorpusTransformation,
)
from embeddings.transformation.flair_transformation.downsample_corpus_transformation import (
    DownsampleFlairCorpusTransformation,
)
from embeddings.transformation.flair_transformation.split_sample_corpus_transformation import (
    SampleSplitsFlairCorpusTransformation,
)
from embeddings.transformation.transformation import Transformation
from embeddings.utils.flair_corpus_persister import FlairConllPersister


@pytest.fixture
def ner_dataset_name() -> str:
    return "clarin-pl/kpwr-ner"


@pytest.fixture
def downsample_percentage() -> float:
    return 0.005


@pytest.fixture
def split_sample_percentage() -> float:
    return 0.1


@pytest.fixture
def ner_data(ner_dataset_name: str) -> datasets.DatasetDict:
    dataset = HuggingFaceDataset(ner_dataset_name)
    data_loader = HuggingFaceDataLoader()
    data = data_loader.load(dataset)
    return data


@pytest.fixture
def ner_downsample_transformation(
    result_path: "TemporaryDirectory[str]",
    downsample_percentage: float,
) -> Transformation[datasets.DatasetDict, Corpus]:
    transformation = (
        ColumnCorpusTransformation("tokens", "ner")
        .then(DownsampleFlairCorpusTransformation(percentage=downsample_percentage, seed=441))
        .persisting(FlairConllPersister(result_path.name))
    )
    return transformation


@pytest.fixture
def ner_split_sample_transformation(
    result_path: "TemporaryDirectory[str]",
    split_sample_percentage: float,
) -> Transformation[datasets.DatasetDict, Corpus]:
    transformation = (
        ColumnCorpusTransformation("tokens", "ner")
        .then(SampleSplitsFlairCorpusTransformation(dev_fraction=split_sample_percentage, seed=441))
        .persisting(FlairConllPersister(result_path.name))
    )
    return transformation


@pytest.fixture
def ner_combined_sample_transformation(
    result_path: "TemporaryDirectory[str]",
    downsample_percentage: float,
    split_sample_percentage: float,
) -> Transformation[datasets.DatasetDict, Corpus]:
    transformation = (
        ColumnCorpusTransformation("tokens", "ner")
        .then(DownsampleFlairCorpusTransformation(percentage=downsample_percentage, seed=441))
        .then(SampleSplitsFlairCorpusTransformation(dev_fraction=split_sample_percentage, seed=441))
        .persisting(FlairConllPersister(result_path.name))
    )
    return transformation


@pytest.fixture
def ner_combined_other_order_sample_transformation(
    result_path: "TemporaryDirectory[str]",
    downsample_percentage: float,
    split_sample_percentage: float,
) -> Transformation[datasets.DatasetDict, Corpus]:
    transformation = (
        ColumnCorpusTransformation("tokens", "ner")
        .then(SampleSplitsFlairCorpusTransformation(dev_fraction=split_sample_percentage, seed=441))
        .then(DownsampleFlairCorpusTransformation(percentage=downsample_percentage, seed=441))
        .persisting(FlairConllPersister(result_path.name))
    )
    return transformation


def test_downsampling(
    result_path: "TemporaryDirectory[str]",
    ner_data: datasets.DatasetDict,
    ner_downsample_transformation: Transformation[datasets.DatasetDict, Corpus],
    downsample_percentage: float,
) -> None:
    flair.set_seed(441)
    expected_train_size = round(len(ner_data["train"]) * downsample_percentage)
    expected_test_size = round(len(ner_data["test"]) * downsample_percentage)

    transformed_data = ner_downsample_transformation.transform(ner_data)

    assert len(transformed_data.train) == expected_train_size
    assert len(transformed_data.test) == expected_test_size
    assert transformed_data.dev is None

    data_loader = ConllFlairCorpusDataLoader()
    dataset = LocalDataset(dataset=result_path.name)
    loaded_data = data_loader.load(dataset)

    assert len(loaded_data.train) == expected_train_size
    assert len(loaded_data.test) == expected_test_size
    assert loaded_data.dev is None

    for sentence_transformed, sentence_loaded in zip(transformed_data.train, loaded_data.train):
        for token_transformed, token_loaded in zip(sentence_transformed, sentence_loaded):
            assert token_transformed.text == token_loaded.text
            assert token_transformed.get_tag("tag") == token_loaded.get_tag("tag")


def test_split_sampling(
    result_path: "TemporaryDirectory[str]",
    ner_data: datasets.DatasetDict,
    ner_split_sample_transformation: Transformation[datasets.DatasetDict, Corpus],
    split_sample_percentage: float,
) -> None:
    flair.set_seed(441)
    expected_train_size = round(len(ner_data["train"]) * (1 - split_sample_percentage))
    expected_dev_size = round(len(ner_data["train"]) * split_sample_percentage)
    expected_test_size = len(ner_data["test"])

    transformed_data = ner_split_sample_transformation.transform(ner_data)

    assert len(transformed_data.train) == expected_train_size
    assert len(transformed_data.dev) == expected_dev_size
    assert len(transformed_data.test) == expected_test_size

    data_loader = ConllFlairCorpusDataLoader()
    dataset = LocalDataset(dataset=result_path.name)
    loaded_data = data_loader.load(dataset)

    assert len(loaded_data.train) == expected_train_size
    assert len(loaded_data.dev) == expected_dev_size
    assert len(loaded_data.test) == expected_test_size

    for sentence_transformed, sentence_loaded in zip(transformed_data.train, loaded_data.train):
        for token_transformed, token_loaded in zip(sentence_transformed, sentence_loaded):
            assert token_transformed.text == token_loaded.text
            assert token_transformed.get_tag("tag") == token_loaded.get_tag("tag")


def test_combined_sampling(
    result_path: "TemporaryDirectory[str]",
    ner_data: datasets.DatasetDict,
    ner_combined_sample_transformation: Transformation[datasets.DatasetDict, Corpus],
    ner_combined_other_order_sample_transformation: Transformation[datasets.DatasetDict, Corpus],
    split_sample_percentage: float,
    downsample_percentage: float,
) -> None:
    flair.set_seed(441)
    expected_train_size = round(
        len(ner_data["train"]) * (1 - split_sample_percentage) * downsample_percentage
    )
    expected_dev_size = round(
        len(ner_data["train"]) * split_sample_percentage * downsample_percentage
    )
    expected_test_size = round(len(ner_data["test"]) * downsample_percentage)

    other_transformed_data = ner_combined_other_order_sample_transformation.transform(
        ner_data.copy()
    )
    transformed_data = ner_combined_sample_transformation.transform(ner_data)

    assert len(transformed_data.train) == expected_train_size
    assert len(transformed_data.dev) == expected_dev_size
    assert len(transformed_data.test) == expected_test_size

    assert len(transformed_data.train) == len(other_transformed_data.train)
    assert len(transformed_data.dev) == len(other_transformed_data.dev)
    assert len(transformed_data.test) == len(other_transformed_data.test)

    data_loader = ConllFlairCorpusDataLoader()
    dataset = LocalDataset(dataset=result_path.name)
    loaded_data = data_loader.load(dataset)

    assert len(loaded_data.train) == expected_train_size
    assert len(loaded_data.dev) == expected_dev_size
    assert len(loaded_data.test) == expected_test_size

    for sentence_transformed, sentence_loaded in zip(transformed_data.train, loaded_data.train):
        for token_transformed, token_loaded in zip(sentence_transformed, sentence_loaded):
            assert token_transformed.text == token_loaded.text
            assert token_transformed.get_tag("tag") == token_loaded.get_tag("tag")
