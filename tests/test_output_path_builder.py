from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import pytest

from embeddings.utils.utils import build_output_path

root = "results/experiment_1"
hub_repo_official = "hub-123"
hub_repo_unofficial = "org/hub-123"


@pytest.fixture(scope="module")
def local_repo() -> Iterator[Path]:
    with TemporaryDirectory() as temp_path:
        local_repo = Path(temp_path).joinpath("local-123")
        local_repo.mkdir()
        yield local_repo


def test_hub_repo_off_unoff() -> None:
    assert (
        str(build_output_path(root, hub_repo_official, hub_repo_unofficial))
        == "results/experiment_1/hub-123/org__hub-123"
    )


def test_hub_repo_off_off() -> None:
    assert (
        str(build_output_path(root, hub_repo_official, hub_repo_official))
        == "results/experiment_1/hub-123/hub-123"
    )


def test_hub_repo_unoff_unoff() -> None:
    assert (
        str(build_output_path(root, hub_repo_unofficial, hub_repo_unofficial))
        == "results/experiment_1/org__hub-123/org__hub-123"
    )


def test_local_repo_hub_str_repo_official(local_repo: Path) -> None:
    assert (
        str(build_output_path(root, local_repo, hub_repo_unofficial))
        == "results/experiment_1/local-123/org__hub-123"
    )


def test_local_repo_path_hub_repo_official(local_repo: Path) -> None:
    assert (
        str(build_output_path(root, Path(local_repo), hub_repo_unofficial))
        == "results/experiment_1/local-123/org__hub-123"
    )


def test_local_repo_path_local_repo_path(local_repo: Path) -> None:
    assert (
        str(build_output_path(root, Path(local_repo), Path(local_repo)))
        == "results/experiment_1/local-123/local-123"
    )


def test_value_error_path_file() -> None:
    with pytest.raises(ValueError):
        build_output_path(root, Path(__file__), hub_repo_unofficial)


def test_value_error_path_notexists() -> None:
    with pytest.raises(ValueError):
        build_output_path(root, Path("not_existing_file"), hub_repo_unofficial)
