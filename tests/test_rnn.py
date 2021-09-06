import flair
import torch
from flair.data import Sentence
from torch.testing import assert_close

from embeddings.embedding.flair_embedding import FlairDocumentRNNEmbeddings
from embeddings.embedding.static.embedding import AutoStaticWordEmbedding


def test_rnn_embedding(dummy_word2vec_config) -> None:
    flair.set_seed(441)

    config = dummy_word2vec_config
    word_embedding = AutoStaticWordEmbedding.from_config(config)
    document_embedding = FlairDocumentRNNEmbeddings(word_embedding)

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
