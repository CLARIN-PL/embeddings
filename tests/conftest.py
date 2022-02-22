from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from gdown import cached_download

from embeddings.embedding.static.fasttext import KGR10FastTextConfig
from embeddings.embedding.static.word2vec import KGR10Word2VecConfig

STATIC_EMBEDDING_URL = (
    "http://dsmodels.nlp.ipipan.waw.pl/dsmodels/wiki-forms-all-100-cbow-ns-30-it100.txt.gz"
)


@pytest.fixture(scope="session")
def local_embedding_filepath() -> Path:
    str_filepath = cached_download(STATIC_EMBEDDING_URL)
    return Path(str_filepath)


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
