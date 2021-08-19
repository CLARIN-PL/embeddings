from typing import Type

import pytest
from flair.embeddings import TransformerDocumentEmbeddings, TransformerWordEmbeddings
from transformers import AlbertModel

from embeddings.embedding.auto_flair import (
    AutoFlairDocumentEmbedding,
    AutoFlairEmbedding,
    AutoFlairWordEmbedding,
)
from embeddings.embedding.flair_embedding import (
    FlairTransformerDocumentEmbedding,
    FlairTransformerWordEmbedding,
)
from embeddings.embedding.static.word import AutoStaticWordEmbedding


def test_automodel_word_transformer() -> None:
    embedding = AutoFlairWordEmbedding.from_hub("hf-internal-testing/tiny-albert")
    assert isinstance(embedding, FlairTransformerWordEmbedding)
    assert isinstance(embedding.model, TransformerWordEmbeddings)
    assert isinstance(embedding.model.model, AlbertModel)


def test_automodel_document_transformer() -> None:
    embedding = AutoFlairDocumentEmbedding.from_hub("hf-internal-testing/tiny-albert")
    assert isinstance(embedding, FlairTransformerDocumentEmbedding)
    assert isinstance(embedding.model, TransformerDocumentEmbeddings)
    assert isinstance(embedding.model.model, AlbertModel)


def check_automodel_error_repo_not_found(embedding: Type[AutoFlairEmbedding]) -> None:
    with pytest.raises(EnvironmentError):
        embedding.from_hub(repo_id="name_of_repo_that_does_not_exist")


def test_automodel_word_error_repo_not_found() -> None:
    check_automodel_error_repo_not_found(AutoFlairWordEmbedding)


def test_automodel_document_error_repo_not_found() -> None:
    check_automodel_error_repo_not_found(AutoFlairDocumentEmbedding)


def check_automodel_error_wrong_format(embedding: Type[AutoFlairEmbedding]) -> None:
    with pytest.raises(EnvironmentError):
        embedding.from_hub(repo_id="sentence-transformers/average_word_embeddings_glove.6B.300d")


def test_automodel_word_error_wrong_format() -> None:
    check_automodel_error_wrong_format(AutoFlairWordEmbedding)


def test_automodel_document_error_wrong_format() -> None:
    check_automodel_error_wrong_format(AutoFlairDocumentEmbedding)


def test_static_automodel_error_repo_not_found() -> None:
    with pytest.raises(EnvironmentError):
        AutoStaticWordEmbedding.from_default_config(repo_id="name_of_repo_that_does_not_exist")


def test_static_automodel_error_wrong_format() -> None:
    with pytest.raises(EnvironmentError):
        AutoStaticWordEmbedding.from_default_config(
            repo_id="sentence-transformers/average_word_embeddings_glove.6B.300d"
        )
