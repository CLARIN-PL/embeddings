from dataclasses import dataclass, field
from typing import Type

import torch
from flair.data import Sentence
from torch.testing import assert_close as assert_close

from embeddings.embedding.static.config import SingleFileConfig
from embeddings.embedding.static.fasttext import KGR10FastTextConfig, KGR10FastTextEmbedding
from embeddings.utils.utils import import_from_string


@dataclass
class DummyFastTextConfig(SingleFileConfig):
    repo_id: str = field(default="clarin-pl/fastText-kgr10", init=False)
    model_name: str = field(default="test/dummy.model.bin", init=False)


def test_fasttext_embeddings_equal() -> None:
    config = DummyFastTextConfig()
    cls: Type[KGR10FastTextEmbedding] = import_from_string(config.type_reference)
    embedding = cls(config)

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


def test_krg10_fasttext_default_config() -> None:
    config = KGR10FastTextConfig()
    assert config.model_name == "kgr10.plain.skipgram.dim300.neg10.bin"
