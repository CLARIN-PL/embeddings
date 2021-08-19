from typing import Type

import pytest
import torch
from flair.data import Sentence
from torch.testing import assert_close as assert_close

from embeddings.embedding.static.fasttext import KGR10FastTextConfig, KGR10FastTextEmbedding
from embeddings.embedding.static.word import AutoStaticWordEmbedding
from embeddings.utils.utils import import_from_string


@pytest.fixture
def dummy_fasttext_config() -> KGR10FastTextConfig:
    config = KGR10FastTextConfig()
    config.model_name = "test/dummy.model.bin"
    return config


def test_fasttext_embeddings_close(dummy_fasttext_config: KGR10FastTextConfig) -> None:
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


def test_static_automodel_fast_text(dummy_fasttext_config: KGR10FastTextConfig) -> None:
    config = dummy_fasttext_config
    embedding = AutoStaticWordEmbedding.from_config(config=config)
    assert isinstance(embedding, KGR10FastTextEmbedding)
    assert_close_embedding(embedding)


# def test_static_automodel_fast_text(dummy_fasttext_config: KGR10FastTextConfig) -> None:
#     config = dummy_fasttext_config
#     embedding = AutoStaticWordEmbedding.from_config(config=config)
#     assert isinstance(embedding, KGR10FastTextEmbedding)
#     assert_close_embedding(embedding)


def assert_close_embedding(embedding: KGR10FastTextEmbedding) -> None:
    sentence = Sentence("Polska należy do Unii Europejskiej.")
    embedding.embed([sentence])

    assert all(token.embedding.shape == (100,) for token in sentence)
    tokens_embeddings = torch.stack([token.embedding for token in sentence])

    assert_close(
        tokens_embeddings[:, :5].clone(),
        torch.Tensor(
            [
                [-1.5600e-03, 7.7694e-03, 4.0358e-03, -3.9865e-03, 1.0068e-02],
                [1.9560e-04, 5.9088e-03, 4.6389e-03, -4.8860e-04, 5.7465e-03],
                [1.5331e-03, 6.9973e-03, -1.2511e-03, -6.6496e-03, 1.7792e-03],
                [-5.3175e-04, 6.8117e-04, 4.8832e-05, -1.9572e-04, 6.4091e-04],
                [-9.3253e-04, 2.3494e-03, 1.8994e-03, -1.3316e-03, 1.6869e-03],
                [9.7278e-03, -1.8942e-03, 6.4082e-03, -4.4753e-03, 1.2305e-02],
            ]
        ),
        atol=1e-6,
        rtol=0,
    )

    torch.testing.assert_close(
        tokens_embeddings.mean(dim=1),
        torch.Tensor([7.9601e-04, 7.0972e-04, 4.8373e-05, 2.1889e-05, 2.7810e-04, 9.9517e-04]),
        atol=1e-6,
        rtol=0,
    )

    torch.testing.assert_close(
        tokens_embeddings.std(dim=1),
        torch.Tensor([0.0045, 0.0029, 0.0036, 0.0019, 0.0020, 0.0058]),
        atol=1e-4,
        rtol=0,
    )
