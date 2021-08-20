import pytest
from flair.embeddings import TransformerWordEmbeddings
from transformers import BertModel

from embeddings.embedding.auto_flair import AutoFlairWordEmbedding
from embeddings.embedding.flair_embedding import FlairTransformerWordEmbedding
from embeddings.embedding.static.word import AutoStaticWordEmbedding


def test_automodel_transformer() -> None:
    embedding = AutoFlairWordEmbedding.from_hub("allegro/herbert-base-cased")
    assert isinstance(embedding, FlairTransformerWordEmbedding)
    assert isinstance(embedding.model, TransformerWordEmbeddings)
    assert isinstance(embedding.model.model, BertModel)


def test_automodel_error_repo_not_found() -> None:
    with pytest.raises(EnvironmentError):
        AutoFlairWordEmbedding.from_hub(repo_id="name_of_repo_that_does_not_exist")


def test_automodel_error_wrong_format() -> None:
    with pytest.raises(EnvironmentError):
        AutoFlairWordEmbedding.from_hub(
            repo_id="sentence-transformers/average_word_embeddings_glove.6B.300d"
        )


def test_static_automodel_error_repo_not_found() -> None:
    with pytest.raises(EnvironmentError):
        AutoStaticWordEmbedding.from_default_config(repo_id="name_of_repo_that_does_not_exist")


def test_static_automodel_error_wrong_format() -> None:
    with pytest.raises(EnvironmentError):
        AutoStaticWordEmbedding.from_default_config(
            repo_id="sentence-transformers/average_word_embeddings_glove.6B.300d"
        )
