from abc import ABC, abstractmethod
from typing import Any, Type, Union

from embeddings.embedding.flair_embedding import FlairDocumentPoolEmbedding, FlairEmbedding
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


class AutoStaticEmbedding(ABC):
    @staticmethod
    @abstractmethod
    def from_default_config(
        repo_id: str, **kwargs: Any
    ) -> Union[StaticWordEmbedding, FlairDocumentPoolEmbedding]:
        pass

    @staticmethod
    @abstractmethod
    def from_config(
        config: StaticModelHubConfig, **kwargs: Any
    ) -> Union[StaticWordEmbedding, FlairDocumentPoolEmbedding]:
        pass


class AutoStaticWordEmbedding(AutoStaticEmbedding):
    @staticmethod
    def from_default_config(repo_id: str, **kwargs: Any) -> StaticWordEmbedding:
        """Returns a static word embedding model initialised using the default_config.json file,
        that should be placed in the given repository.
        Example: https://huggingface.co/clarin-pl/fastText-kgr10/blob/main/default_config.json"""
        config = StaticModelHubConfig(repo_id)
        return AutoStaticWordEmbedding._get_model_cls(config).from_default_config(config, **kwargs)

    @staticmethod
    def from_config(config: StaticModelHubConfig, **kwargs: Any) -> StaticWordEmbedding:
        """Returns a static word embedding model initialised using the provided config."""
        return AutoStaticWordEmbedding._get_model_cls(config).from_config(config, **kwargs)

    @staticmethod
    def _get_model_cls(config: StaticModelHubConfig) -> Type[StaticWordEmbedding]:
        """Return a class of a static word embedding model. The class should be specified in the
        module.json file, that should be placed in the given repository.
        Example: https://huggingface.co/clarin-pl/fastText-kgr10/blob/main/module.json"""
        cls: Type[StaticWordEmbedding] = import_from_string(config.model_type_reference)
        if not issubclass(cls, StaticWordEmbedding):
            raise ValueError(f"Type reference has to be subclass of {StaticWordEmbedding}.")
        return cls


class AutoStaticDocumentEmbedding(AutoStaticEmbedding):
    @staticmethod
    def from_default_config(repo_id: str, **kwargs: Any) -> FlairDocumentPoolEmbedding:
        """Returns a static word embedding model initialised using the default_config.json file,
        that should be placed in the given repository.
        Example: https://huggingface.co/clarin-pl/fastText-kgr10/blob/main/default_config.json"""
        embedding = AutoStaticWordEmbedding.from_default_config(repo_id, **kwargs)
        return FlairDocumentPoolEmbedding(embedding)

    @staticmethod
    def from_config(config: StaticModelHubConfig, **kwargs: Any) -> FlairDocumentPoolEmbedding:
        """Returns a static word embedding model initialised using the provided config."""
        embedding = AutoStaticWordEmbedding.from_config(config, **kwargs)
        return FlairDocumentPoolEmbedding(embedding)
