from abc import ABC, abstractmethod
from typing import Any, Optional, Type

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

    @staticmethod
    @abstractmethod
    def from_config(config: StaticModelHubConfig, **kwargs: Any) -> "StaticWordEmbedding":
        pass


class AutoStaticWordEmbedding:
    @staticmethod
    def from_hub(
        repo_id: Optional[str] = None, config: Optional[StaticModelHubConfig] = None, **kwargs: Any
    ) -> StaticWordEmbedding:
        if not config and repo_id:
            config = StaticModelHubConfig(repo_id)
            return AutoStaticWordEmbedding._get_model_cls(config).from_default_config(
                config, **kwargs
            )
        elif not repo_id and config:
            return AutoStaticWordEmbedding._get_model_cls(config).from_config(config, **kwargs)
        else:
            raise ValueError("You should pass repo_id or config, not both.")

    @staticmethod
    def _get_model_cls(config: StaticModelHubConfig) -> Type[StaticWordEmbedding]:
        cls: Type[StaticWordEmbedding] = import_from_string(config.model_type_reference)
        if not issubclass(cls, StaticWordEmbedding):
            raise ValueError(f"Type reference has to be subclass of {StaticWordEmbedding}.")
        return cls
