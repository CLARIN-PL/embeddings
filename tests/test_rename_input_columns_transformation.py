import pytest

from embeddings.data.data_loader import HuggingFaceDataLoader
from embeddings.data.dataset import Dataset
from embeddings.transformation.hf_transformation.to_pandas_transformation import (
    ToPandasHuggingFaceCorpusTransformation,
)
from embeddings.transformation.pandas_transformation.rename_input_columns_transformation import (
    RenameInputColumnsTransformation,
)


@pytest.fixture(scope="module")
def hf_dataset(dataset_name: str = "clarin-pl/polemo2-official"):
    return Dataset(dataset_name)


@pytest.fixture(scope="module")
def data_loader():
    return HuggingFaceDataLoader()


@pytest.fixture(scope="module")
def rename_columns_transformation(
    input_column_name: str = "text", target_column_name: str = "target"
):
    return ToPandasHuggingFaceCorpusTransformation().then(
        RenameInputColumnsTransformation(input_column_name, target_column_name)
    )


def test_rename_input_columns_transformation(
    hf_dataset: Dataset,
    data_loader: HuggingFaceDataLoader,
    rename_columns_transformation: ToPandasHuggingFaceCorpusTransformation,
):
    dataset = hf_dataset
    loader = data_loader
    transformation = rename_columns_transformation

    data = loader.load(dataset)
    transformed_data = transformation.transform(data)

    assert "x" in transformed_data["train"].columns
    assert "x" in transformed_data["validation"].columns
    assert "x" in transformed_data["test"].columns
    assert "y" in transformed_data["train"].columns
    assert "y" in transformed_data["validation"].columns
    assert "y" in transformed_data["test"].columns
