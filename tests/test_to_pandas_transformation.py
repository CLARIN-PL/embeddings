import pytest

from embeddings.data.data_loader import HuggingFaceDataLoader
from embeddings.data.dataset import Dataset
from embeddings.transformation.hf_transformation.to_pandas_transformation import (
    ToPandasHuggingFaceCorpusTransformation,
)


@pytest.fixture(scope="module")
def hf_dataset(dataset_name: str = "clarin-pl/polemo2-official") -> Dataset:
    return Dataset(dataset_name)


@pytest.fixture(scope="module")
def data_loader() -> HuggingFaceDataLoader:
    return HuggingFaceDataLoader()


@pytest.fixture(scope="module")
def to_pandas_transformation() -> ToPandasHuggingFaceCorpusTransformation:
    return ToPandasHuggingFaceCorpusTransformation()


def test_to_pandas_transformation(
    hf_dataset: Dataset,
    data_loader: HuggingFaceDataLoader,
    to_pandas_transformation: ToPandasHuggingFaceCorpusTransformation,
) -> None:
    dataset = hf_dataset
    loader = data_loader
    transformation = to_pandas_transformation

    data = loader.load(dataset)
    transformed_data = transformation.transform(data)

    assert "train" in transformed_data.keys()
    assert "validation" in transformed_data.keys()
    assert "test" in transformed_data.keys()
    assert "text" in transformed_data["train"].columns
    assert "text" in transformed_data["validation"].columns
    assert "text" in transformed_data["test"].columns
    assert "target" in transformed_data["train"].columns
    assert "target" in transformed_data["validation"].columns
    assert "target" in transformed_data["test"].columns
    assert transformed_data["train"].shape[0] == 6573
    assert transformed_data["train"].shape[1] == 2
    assert transformed_data["validation"].shape[0] == 823
    assert transformed_data["validation"].shape[1] == 2
    assert transformed_data["test"].shape[0] == 820
    assert transformed_data["test"].shape[1] == 2
