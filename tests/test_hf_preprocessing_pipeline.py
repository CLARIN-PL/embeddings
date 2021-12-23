import math
from copy import deepcopy

import datasets
import pytest

from embeddings.data.data_loader import HuggingFaceDataLoader
from embeddings.data.dataset import HuggingFaceDataset
from embeddings.transformation.hf_transformation.drop_subset_transformation import (
    DropSubsetHuggingFaceCorpusTransformation,
)
from embeddings.transformation.hf_transformation.sampling_transformation import (
    SampleSplitsHuggingFaceTransformation,
)


@pytest.fixture
def hf_dataset_cfg():
    return {
        "dataset": "clarin-pl/polemo2-official",
    }


@pytest.fixture
def hf_dataset(hf_dataset_cfg):
    return HuggingFaceDataLoader().load(HuggingFaceDataset(**hf_dataset_cfg))


def test_hf_dataset(hf_dataset):
    assert isinstance(hf_dataset, datasets.DatasetDict)


def test_drop_subset_huggingface_corpus_transformation(hf_dataset):
    transformation = DropSubsetHuggingFaceCorpusTransformation(subset="test")
    transformed_ds = transformation.transform(deepcopy(hf_dataset))
    assert frozenset(transformed_ds.keys()) == frozenset(("train", "validation"))


def test_sample_splits_huggingface_transformation_with_existing_keys(hf_dataset):
    transformation = SampleSplitsHuggingFaceTransformation(dev_fraction=0.2, test_fraction=0.1)
    transfomed_ds = transformation.transform(hf_dataset)
    assert transfomed_ds == hf_dataset


def test_sample_splits_huggingface_transformation_with_non_existing_keys(hf_dataset):
    transformation = (
        DropSubsetHuggingFaceCorpusTransformation(subset="test")
        .then(DropSubsetHuggingFaceCorpusTransformation(subset="validation"))
        .then(SampleSplitsHuggingFaceTransformation(dev_fraction=0.2, test_fraction=0.1))
    )
    transfomed_ds = transformation.transform(deepcopy(hf_dataset))
    assert math.floor(len(hf_dataset["train"]) * 0.7) == len(transfomed_ds["train"])
    assert math.floor(len(hf_dataset["train"]) * 0.2) == len(transfomed_ds["validation"])
    assert math.ceil(len(hf_dataset["train"]) * 0.1) == len(transfomed_ds["test"])
