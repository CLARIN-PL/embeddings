from abc import ABC
from typing import Any

from flair.embeddings import WordEmbeddings

from embeddings.embedding.flair_embedding import FlairEmbedding
from embeddings.embedding.static.config import SingleFileConfig
from embeddings.embedding.static.word import StaticWordEmbedding


class SingleFileEmbedding(StaticWordEmbedding, ABC):
    def __init__(self, config: SingleFileConfig, **load_model_kwargs: Any):
        super().__init__(config.cached_model, **load_model_kwargs)
        self.config = config


class StandardStaticWordEmbedding(FlairEmbedding):
    def _get_model(self) -> WordEmbeddings:
        return WordEmbeddings(self.name, **self.load_model_kwargs)
