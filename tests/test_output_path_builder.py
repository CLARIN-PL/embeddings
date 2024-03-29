from pathlib import Path

import pytest
from _pytest.tmpdir import TempdirFactory

from embeddings.utils.utils import build_output_path

root = "results/experiment_1"
hub_repo_official = "hub-123"
hub_repo_unofficial = "org/hub-123"


@pytest.fixture(scope="module")
def local_repo(tmpdir_factory: TempdirFactory) -> Path:
    local_repo = tmpdir_factory.mktemp("tmp").join("local-123")
    local_repo.mkdir()
    return Path(local_repo)


def test_hub_repo_off_unoff() -> None:
    assert str(
        build_output_path(
            root, hub_repo_official, hub_repo_unofficial, timestamp_subdir=False, mkdirs=False
        )
    ) == str(Path("results/experiment_1/hub-123/org__hub-123"))


def test_hub_repo_off_off() -> None:
    assert str(
        build_output_path(
            root, hub_repo_official, hub_repo_official, timestamp_subdir=False, mkdirs=False
        )
    ) == str(Path("results/experiment_1/hub-123/hub-123"))


def test_hub_repo_unoff_unoff() -> None:
    assert str(
        build_output_path(
            root, hub_repo_unofficial, hub_repo_unofficial, timestamp_subdir=False, mkdirs=False
        )
    ) == str(Path("results/experiment_1/org__hub-123/org__hub-123"))


def test_local_repo_hub_str_repo_official(local_repo: Path) -> None:
    assert str(
        build_output_path(
            root, local_repo, hub_repo_unofficial, timestamp_subdir=False, mkdirs=False
        )
    ) == str(Path("results/experiment_1/local-123/org__hub-123"))


def test_local_repo_path_hub_repo_official(local_repo: Path) -> None:
    assert str(
        build_output_path(
            root, Path(local_repo), hub_repo_unofficial, timestamp_subdir=False, mkdirs=False
        )
    ) == str(Path("results/experiment_1/local-123/org__hub-123"))


def test_local_repo_path_local_repo_path(local_repo: Path) -> None:
    assert str(
        build_output_path(
            root, Path(local_repo), Path(local_repo), timestamp_subdir=False, mkdirs=False
        )
    ) == str(Path("results/experiment_1/local-123/local-123"))


def test_value_error_path_file() -> None:
    with pytest.raises(ValueError):
        build_output_path(
            root, Path(__file__), hub_repo_unofficial, timestamp_subdir=False, mkdirs=False
        )


def test_value_error_path_notexists() -> None:
    with pytest.raises(ValueError):
        build_output_path(
            root,
            Path("not_existing_file"),
            hub_repo_unofficial,
            timestamp_subdir=False,
            mkdirs=False,
        )
