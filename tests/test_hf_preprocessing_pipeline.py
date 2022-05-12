import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import datasets
import pytest

from embeddings.data.data_loader import HuggingFaceDataLoader, HuggingFaceLocalDataLoader
from embeddings.data.dataset import Dataset
from embeddings.pipeline.hf_preprocessing_pipeline import HuggingFacePreprocessingPipeline
from embeddings.transformation.hf_transformation.drop_subset_transformation import (
    DropSubsetHuggingFaceCorpusTransformation,
)
from embeddings.transformation.hf_transformation.sampling_transformation import (
    SampleSplitsHuggingFaceTransformation,
)


@pytest.fixture(scope="module")
def hf_dataset_cfg() -> Dict[str, str]:
    return {
        "dataset": "clarin-pl/polemo2-official",
    }


@pytest.fixture(scope="module")
def hf_preprocessing_pipeline_kwargs(
    hf_dataset_cfg: Dict[str, str], result_path: Path
) -> Dict[str, Any]:
    return {
        "dataset_name": hf_dataset_cfg["dataset"],
        "persist_path": result_path.name,
        "sample_missing_splits": None,
        "ignore_test_subset": True,
        "seed": 441,
        "load_dataset_kwargs": None,
    }


@pytest.fixture(scope="module")
def hf_dataset(hf_dataset_cfg: Dict[str, str]) -> datasets.DatasetDict:
    return HuggingFaceDataLoader().load(Dataset(**hf_dataset_cfg))


def test_hf_dataset(hf_dataset: datasets.DatasetDict) -> None:
    assert isinstance(hf_dataset, datasets.DatasetDict)


def test_drop_subset_huggingface_corpus_transformation(hf_dataset: datasets.DatasetDict) -> None:
    transformation = DropSubsetHuggingFaceCorpusTransformation(subset="test")
    transformed_ds = transformation.transform(deepcopy(hf_dataset))
    assert frozenset(transformed_ds.keys()) == frozenset(("train", "validation"))


def test_sample_splits_huggingface_transformation_with_existing_keys(
    hf_dataset: datasets.DatasetDict,
) -> None:
    transformation = SampleSplitsHuggingFaceTransformation(dev_fraction=0.2, test_fraction=0.1)
    transfomed_ds = transformation.transform(hf_dataset)
    assert transfomed_ds == hf_dataset


def test_sample_splits_huggingface_transformation_with_non_existing_keys(
    hf_dataset: datasets.DatasetDict,
) -> None:
    transformation = (
        DropSubsetHuggingFaceCorpusTransformation(subset="test")
        .then(DropSubsetHuggingFaceCorpusTransformation(subset="validation"))
        .then(SampleSplitsHuggingFaceTransformation(dev_fraction=0.2, test_fraction=0.1))
    )
    transfomed_ds = transformation.transform(deepcopy(hf_dataset))
    assert math.floor(len(hf_dataset["train"]) * 0.7) == len(transfomed_ds["train"])
    assert math.floor(len(hf_dataset["train"]) * 0.2) == len(transfomed_ds["validation"])
    assert math.ceil(len(hf_dataset["train"]) * 0.1) == len(transfomed_ds["test"])


def test_huggingface_text_classification_preprocessing_pipeline(
    hf_preprocessing_pipeline_kwargs: Dict[str, Any]
) -> None:
    pipeline = HuggingFacePreprocessingPipeline(**hf_preprocessing_pipeline_kwargs)
    pipeline.run()

    dataset = HuggingFaceLocalDataLoader().load(
        Dataset(hf_preprocessing_pipeline_kwargs["persist_path"])
    )
    assert isinstance(dataset, datasets.DatasetDict)
    assert frozenset(dataset.keys()) == frozenset(("train", "validation"))
