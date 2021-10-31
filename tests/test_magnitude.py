from numpy import ndarray

from embeddings.embedding.static.magnitude import AutoMagnitude, MagnitudeEmbedding


def test_magnitude() -> None:
    embedding = AutoMagnitude.from_url(
        "http://magnitude.plasticity.ai/glove/light/glove.6B.50d.magnitude"
    )
    assert isinstance(embedding, MagnitudeEmbedding)
    assert isinstance(embedding.embed(["test"]), ndarray)
