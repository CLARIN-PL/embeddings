from pathlib import Path
from typing import Type

import pytest
import torch
from flair.data import Sentence
from torch.testing import assert_close

from embeddings.embedding.static.embedding import (
    AutoStaticWordEmbedding,
    LocalFileAutoStaticWordEmbedding,
)
from embeddings.embedding.static.word2vec import (
    IPIPANWord2VecConfig,
    IPIPANWord2VecEmbedding,
    KGR10Word2VecConfig,
    KGR10Word2VecEmbedding,
)
from embeddings.utils.utils import import_from_string


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


def test_available_config_model_1() -> None:
    KGR10Word2VecConfig(method="cbow", hs=True, dimension=300, mwe=True)


def test_available_config_model_2() -> None:
    KGR10Word2VecConfig(method="skipgram", hs=False, dimension=300, mwe=True)


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


@pytest.fixture()
def ipipan_embedding_config(local_embedding_filepath: Path) -> IPIPANWord2VecConfig:
    return IPIPANWord2VecConfig(local_embedding_filepath)


def test_ipipan_default_config(ipipan_embedding_config: IPIPANWord2VecConfig) -> None:
    assert ipipan_embedding_config.model_name == "wiki-forms-all-100-cbow-ns-30-it100"


def test_ipipan_word2vec_embeddings_equal(ipipan_embedding_config: IPIPANWord2VecConfig) -> None:
    config = ipipan_embedding_config
    cls: Type[IPIPANWord2VecEmbedding] = import_from_string(config.model_type_reference)
    embedding = cls(str(config.model_file_path))
    assert_close_ipipan_embedding(embedding)


def test_init_ipipan_word2vec_from_file(local_embedding_filepath: Path) -> None:
    embedding = IPIPANWord2VecEmbedding.from_file(local_embedding_filepath)
    assert_close_ipipan_embedding(embedding)


def test_static_automodel_ipipan_word2vec(
    local_embedding_filepath: Path,
    model_type_reference: str = "embeddings.embedding.static.word2vec.IPIPANWord2VecEmbedding",
) -> None:
    embedding = LocalFileAutoStaticWordEmbedding.from_file(
        local_embedding_filepath, model_type_reference
    )
    assert isinstance(embedding, IPIPANWord2VecEmbedding)
    assert_close_ipipan_embedding(embedding)


def assert_close_ipipan_embedding(embedding: IPIPANWord2VecEmbedding) -> None:
    sentence = Sentence("Nie zmniejszyło mienności. Wietnam.")
    embedding.embed([sentence])

    assert all(token.embedding.shape == (100,) for token in sentence)
    tokens_embeddings = torch.stack([token.embedding for token in sentence])

    assert_close(
        tokens_embeddings[:, :5].clone(),
        torch.Tensor(
            [
                [-1.5715, -2.4608, -0.8111, -3.2904, -3.7435],
                [-0.8453, -1.3651, 1.3142, -1.0198, -0.0404],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [3.9183, -3.3546, 1.7465, 0.9443, 1.9432],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            ]
        ),
        atol=1e-4,
        rtol=0,
    )

    torch.testing.assert_close(
        tokens_embeddings.mean(dim=1),
        torch.Tensor([0.3991, -0.5544, 0.0000, 0.0000, 0.2900, 0.0000]),
        atol=1e-4,
        rtol=0,
    )

    torch.testing.assert_close(
        tokens_embeddings.std(dim=1),
        torch.Tensor([4.3554, 3.4393, 0.0000, 0.0000, 2.7211, 0.0000]),
        atol=1e-4,
        rtol=0,
    )
