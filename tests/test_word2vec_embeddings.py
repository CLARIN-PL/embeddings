from typing import Type

import pytest
import torch
from flair.data import Sentence
from torch.testing import assert_close

from embeddings.embedding.static.embedding import AutoStaticWordEmbedding
from embeddings.embedding.static.word2vec import KGR10Word2VecConfig, KGR10Word2VecEmbedding
from embeddings.utils.utils import import_from_string


@pytest.fixture
def dummy_word2vec_config() -> KGR10Word2VecConfig:
    config = KGR10Word2VecConfig()
    config.model_name = "test/dummy.model.gensim"
    return config


def test_default_config() -> None:
    assert KGR10Word2VecConfig().model_name == "cbow.v300.m8.hs.mwe.w2v.gensim"


def test_word2vec_embeddings_equal(dummy_word2vec_config: KGR10Word2VecConfig) -> None:
    config = dummy_word2vec_config
    cls: Type[KGR10Word2VecEmbedding] = import_from_string(config.model_type_reference)
    embedding = cls(config)
    assert_close_embedding(embedding)


def test_init_kgr10_word2vec_from_config(dummy_word2vec_config: KGR10Word2VecConfig) -> None:
    config = dummy_word2vec_config
    embedding = KGR10Word2VecEmbedding.from_config(config)
    assert_close_embedding(embedding)


def test_static_automodel_word2vec(dummy_word2vec_config: KGR10Word2VecConfig) -> None:
    config = dummy_word2vec_config
    embedding = AutoStaticWordEmbedding.from_config(config=config)
    assert isinstance(embedding, KGR10Word2VecEmbedding)
    assert_close_embedding(embedding)


def test_not_available_config() -> None:
    with pytest.raises(ValueError):
        KGR10Word2VecConfig(method="cbow", hs=False)


def assert_close_embedding(embedding: KGR10Word2VecEmbedding) -> None:
    sentence = Sentence("Nie zmniejszyło mienności. Wietnam.")
    embedding.embed([sentence])

    assert all(token.embedding.shape == (300,) for token in sentence)
    tokens_embeddings = torch.stack([token.embedding for token in sentence])

    assert_close(
        tokens_embeddings[:, :5].clone(),
        torch.Tensor(
            [
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [-0.0347, -0.0354, 0.0054, 0.0371, 0.1452],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [-0.0923, -0.1034, 0.0055, -0.0999, -0.0180],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            ]
        ),
        atol=1e-4,
        rtol=0,
    )

    torch.testing.assert_close(
        tokens_embeddings.mean(dim=1),
        torch.Tensor([0.0000e00, 0.0000e00, 1.5789e-03, 0.0000e00, 5.7181e-05, 0.0000e00]),
        atol=1e-6,
        rtol=0,
    )

    torch.testing.assert_close(
        tokens_embeddings.std(dim=1),
        torch.Tensor([0.0000, 0.0000, 0.0578, 0.0000, 0.0578, 0.0000]),
        atol=1e-4,
        rtol=0,
    )
