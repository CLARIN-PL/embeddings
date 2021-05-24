from tempfile import TemporaryDirectory

import pytest


@pytest.fixture(scope="session")  # type: ignore
def result_path() -> "TemporaryDirectory[str]":
    return TemporaryDirectory()
