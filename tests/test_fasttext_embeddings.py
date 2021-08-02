from typing import Type

import pytest
import torch
from flair.data import Sentence
from torch.testing import assert_close as assert_close

from embeddings.embedding.auto_flair import AutoWordEmbedding
from embeddings.embedding.static.fasttext import KGR10FastTextConfig, KGR10FastTextEmbedding
from embeddings.embedding.static.word import AutoStaticWordEmbedding
from embeddings.utils.utils import import_from_string


@pytest.fixture
def dummy_fasttext_config() -> KGR10FastTextConfig:
    config = KGR10FastTextConfig()
    config.model_name = "test/dummy.model.bin"
    return config


def assert_close_embedding(embedding: KGR10FastTextEmbedding) -> None:
    sentence = Sentence("że Urban humor życie")
    embedding.embed([sentence])

    assert all(token.embedding.shape == (100,) for token in sentence)
    tokens_embeddings = torch.stack([token.embedding for token in sentence])

    assert_close(
        tokens_embeddings[:, :5].clone(),
        torch.Tensor(
            [
                [-1.8270385e-03, -5.9842765e-03, -3.8310357e-03, -7.0242281e-04, -8.2756989e-03],
                [4.7445842e-03, 1.2122805e-02, 4.5219976e-03, -4.7907191e-03, 1.3309082e-03],
                [7.0265663e-04, -1.4079176e-04, -6.0826674e-04, 4.3734832e-04, 1.3302111e-03],
                [9.5479144e-04, -5.9049390e-04, -9.5329982e-05, 1.7576268e-03, -3.2415607e-03],
            ]
        ),
        atol=1e-7,
        rtol=0,
    )

    torch.testing.assert_close(
        tokens_embeddings.mean(dim=1),
        torch.Tensor([-2.8173163e-04, 8.6010856e-05, -4.0680075e-05, 3.2494412e-04]),
        atol=1e-7,
        rtol=0,
    )

    torch.testing.assert_close(
        tokens_embeddings.std(dim=1),
        torch.Tensor([0.0037086, 0.0046605, 0.0014699, 0.0016064]),
        atol=1e-7,
        rtol=0,
    )


def test_fasttext_embeddings_equal(dummy_fasttext_config: KGR10FastTextConfig) -> None:
    config = dummy_fasttext_config
    cls: Type[KGR10FastTextEmbedding] = import_from_string(config.model_type_reference)
    embedding = cls(config)
    assert_close_embedding(embedding)


def test_krg10_fasttext_default_config() -> None:
    config = KGR10FastTextConfig()
    assert config.model_name == "kgr10.plain.skipgram.dim300.neg10.bin"


def test_init_kgr10_fasttext_from_config(dummy_fasttext_config: KGR10FastTextConfig) -> None:
    config = dummy_fasttext_config
    embedding = KGR10FastTextEmbedding.from_config(config)
    assert_close_embedding(embedding)


def test_automodel_passing_both_args(dummy_fasttext_config: KGR10FastTextConfig) -> None:
    config = dummy_fasttext_config
    with pytest.raises(ValueError):
        AutoStaticWordEmbedding.from_hub(repo_id="test", config=config)


def test_static_automodel_fast_text(dummy_fasttext_config: KGR10FastTextConfig) -> None:
    config = dummy_fasttext_config
    embedding = AutoStaticWordEmbedding.from_hub(config=config)
    assert isinstance(embedding, KGR10FastTextEmbedding)
    assert_close_embedding(embedding)


def test_automodel_fast_text(dummy_fasttext_config: KGR10FastTextConfig) -> None:
    config = dummy_fasttext_config
    embedding = AutoWordEmbedding.from_hub(config=config)
    assert isinstance(embedding, KGR10FastTextEmbedding)
    assert_close_embedding(embedding)
