from unittest.mock import MagicMock

import pytest
from flair.embeddings import TransformerWordEmbeddings
from transformers import BertModel

from embeddings.embedding.auto_flair import AutoFlairWordEmbedding
from embeddings.embedding.flair_embedding import FlairTransformerWordEmbedding
from embeddings.embedding.static.config import StaticModelHubConfig


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


def test_automodel_fast_text_error() -> None:
    config = MagicMock(spec=StaticModelHubConfig)
    with pytest.raises(ValueError):
        AutoFlairWordEmbedding.from_hub(repo_id="repo_name", config=config)
