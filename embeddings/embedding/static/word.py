from abc import ABC, abstractmethod
from typing import Any, Type

from embeddings.embedding.flair_embedding import FlairEmbedding
from embeddings.embedding.static.config import StaticModelHubConfig
from embeddings.utils.utils import import_from_string


class StaticWordEmbedding(FlairEmbedding, ABC):
    @staticmethod
    @abstractmethod
    def get_config(**kwargs: Any) -> StaticModelHubConfig:
        pass

    @staticmethod
    @abstractmethod
    def from_default_config(config: StaticModelHubConfig, **kwargs: Any) -> "StaticWordEmbedding":
        pass


class AutoStaticWordEmbedding:
    @staticmethod
    def from_hub(repo_id: str, **kwargs: Any) -> StaticWordEmbedding:
        config = StaticModelHubConfig(repo_id)
        embedding_class: Type[StaticWordEmbedding] = import_from_string(config.type_reference)
        return embedding_class.from_default_config(config, **kwargs)
