from tempfile import TemporaryDirectory

import pytest

from embeddings.embedding.static.word2vec import KGR10Word2VecConfig


@pytest.fixture(scope="session")
def result_path() -> "TemporaryDirectory[str]":
    return TemporaryDirectory()


@pytest.fixture(scope="session")
def dummy_word2vec_config() -> KGR10Word2VecConfig:
    config = KGR10Word2VecConfig()
    config.model_name = "test/dummy.model.gensim"
    return config
