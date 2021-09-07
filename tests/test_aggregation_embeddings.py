import flair
import pytest
import torch
from flair.data import Sentence
from torch.testing import assert_close

from embeddings.embedding.flair_embedding import (
    FlairDocumentCNNEmbeddings,
    FlairDocumentRNNEmbeddings,
)
from embeddings.embedding.static.embedding import AutoStaticWordEmbedding, StaticEmbedding
from embeddings.embedding.static.word2vec import KGR10Word2VecConfig


@pytest.fixture
def dummy_word2vec(dummy_word2vec_config: KGR10Word2VecConfig) -> StaticEmbedding:
    config = dummy_word2vec_config
    word_embedding = AutoStaticWordEmbedding.from_config(config)
    return word_embedding


def test_rnn_embedding(dummy_word2vec: StaticEmbedding) -> None:
    flair.set_seed(441)
    document_embedding = FlairDocumentRNNEmbeddings(dummy_word2vec)

    sentence = Sentence("Nie zmniejszyło mienności. Wietnam.")
    assert sentence.embedding.shape == (0,)

    with torch.no_grad():
        document_embedding.embed([sentence])

    assert sentence.embedding.shape == (128,)
    assert_close(
        sentence.embedding[:5].clone(),
        torch.Tensor([-0.0532, -0.0119, -0.0310, -0.0637, 0.0989]),
        atol=1e-4,
        rtol=0,
    )

    assert_close(
        sentence.embedding.mean().item(),
        -0.0040,
        atol=1e-4,
        rtol=0,
    )

    assert_close(
        sentence.embedding.std().item(),
        0.0648,
        atol=1e-4,
        rtol=0,
    )


def test_cnn_embedding(dummy_word2vec: StaticEmbedding) -> None:
    flair.set_seed(441)
    document_embedding = FlairDocumentCNNEmbeddings(dummy_word2vec)

    sentence = Sentence("Nie zmniejszyło mienności. Wietnam.")
    assert sentence.embedding.shape == (0,)

    with torch.no_grad():
        document_embedding.embed([sentence])

    assert sentence.embedding.shape == (300,)
    assert_close(
        sentence.embedding[:5].clone(),
        torch.Tensor([0.0000, 0.0252, 0.0082, 0.0353, 0.0391]),
        atol=1e-4,
        rtol=0,
    )

    assert_close(
        sentence.embedding.mean().item(),
        0.0234,
        atol=1e-4,
        rtol=0,
    )

    assert_close(
        sentence.embedding.std().item(),
        0.0232,
        atol=1e-4,
        rtol=0,
    )
