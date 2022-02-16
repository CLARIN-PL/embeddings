import gzip
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from embeddings.embedding.static.fasttext import KGR10FastTextConfig
from embeddings.embedding.static.word2vec import KGR10Word2VecConfig
from embeddings.utils.utils import download_file

STATIC_EMBEDDING_URL = (
    "http://dsmodels.nlp.ipipan.waw.pl/dsmodels/wiki-forms-all-100-cbow-ns-30-it100.txt.gz"
)
STATIC_EMBEDDING_PATH = Path("tests/models/wiki-forms-all-100-cbow-ns-30-it100.txt")


def pytest_sessionstart():
    if not os.path.exists(STATIC_EMBEDDING_PATH.parent):
        os.mkdir(STATIC_EMBEDDING_PATH.parent)

    if not os.path.exists(STATIC_EMBEDDING_PATH):
        tmp_file, _ = download_file(STATIC_EMBEDDING_URL)
        with gzip.open(tmp_file.name, "rb") as f_in:
            with open(STATIC_EMBEDDING_PATH, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)


@pytest.fixture(scope="session")
def result_path() -> "TemporaryDirectory[str]":
    return TemporaryDirectory()


@pytest.fixture(scope="session")
def dummy_word2vec_config() -> KGR10Word2VecConfig:
    config = KGR10Word2VecConfig()
    config.model_name = "test/dummy.model.gensim"
    return config


@pytest.fixture(scope="session")
def dummy_fasttext_config() -> KGR10FastTextConfig:
    config = KGR10FastTextConfig()
    config.model_name = "test/dummy.model.bin"
    return config


def pytest_configure() -> None:
    pytest.decimal = 3
