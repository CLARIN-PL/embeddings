from tempfile import TemporaryDirectory

import pytest

STATIC_EMBEDDING_URL = (
    "http://dsmodels.nlp.ipipan.waw.pl/dsmodels/wiki-forms-all-100-cbow-ns-30-it100.txt.gz"
)


@pytest.fixture(scope="session")
def result_path() -> "TemporaryDirectory[str]":
    return TemporaryDirectory()


def pytest_configure() -> None:
    pytest.decimal = 3
