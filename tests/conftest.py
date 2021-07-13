from tempfile import TemporaryDirectory

import pytest


@pytest.fixture(scope="session")
def result_path() -> "TemporaryDirectory[str]":
    return TemporaryDirectory()
