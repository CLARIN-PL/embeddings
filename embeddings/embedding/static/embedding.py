from abc import ABC, abstractmethod
from typing import Any, Type, Union

from flair.embeddings import WordEmbeddings

from embeddings.embedding.flair_embedding import FlairDocumentPoolEmbedding, FlairEmbedding
from embeddings.embedding.static.config import SingleFileConfig, StaticModelHubConfig
from embeddings.utils.utils import import_from_string


class StaticEmbedding(FlairEmbedding, ABC):
    @staticmethod
    @abstractmethod
    def get_config(**kwargs: Any) -> StaticModelHubConfig:
        pass

    @staticmethod
    @abstractmethod
    def from_default_config(config: StaticModelHubConfig, **kwargs: Any) -> "StaticEmbedding":
        pass

    @staticmethod
    @abstractmethod
    def from_config(config: StaticModelHubConfig, **kwargs: Any) -> "StaticEmbedding":
        pass


class SingleFileEmbedding(StaticEmbedding, ABC):
    def __init__(self, config: SingleFileConfig, **load_model_kwargs: Any):
        super().__init__(config.cached_model, **load_model_kwargs)
        self.config = config


class StandardStaticWordEmbedding(FlairEmbedding):
    def _get_model(self) -> WordEmbeddings:
        return WordEmbeddings(self.name, **self.load_model_kwargs)


class AutoStaticEmbedding(ABC):
    @staticmethod
    @abstractmethod
    def from_default_config(
        repo_id: str, **kwargs: Any
    ) -> Union[StaticEmbedding, FlairDocumentPoolEmbedding]:
        """Returns a static embedding model initialised using the default_config.json file,
        that should be placed in the given repository.
        Example: https://huggingface.co/clarin-pl/fastText-kgr10/blob/main/default_config.json"""
        pass

    @staticmethod
    @abstractmethod
    def from_config(
        config: StaticModelHubConfig, **kwargs: Any
    ) -> Union[StaticEmbedding, FlairDocumentPoolEmbedding]:
        """Returns a static embedding model initialised using the provided config."""
        pass


class AutoStaticWordEmbedding(AutoStaticEmbedding):
    @staticmethod
    def from_default_config(repo_id: str, **kwargs: Any) -> StaticEmbedding:
        config = StaticModelHubConfig(repo_id)
        return AutoStaticWordEmbedding._get_model_cls(config).from_default_config(config, **kwargs)

    @staticmethod
    def from_config(config: StaticModelHubConfig, **kwargs: Any) -> StaticEmbedding:
        return AutoStaticWordEmbedding._get_model_cls(config).from_config(config, **kwargs)

    @staticmethod
    def _get_model_cls(config: StaticModelHubConfig) -> Type[StaticEmbedding]:
        """Returns a class of a static word embedding model. The class should be specified in the
        module.json file, that should be placed in the given repository.
        Example: https://huggingface.co/clarin-pl/fastText-kgr10/blob/main/module.json"""
        cls: Type[StaticEmbedding] = import_from_string(config.model_type_reference)
        if not issubclass(cls, StaticEmbedding):
            raise ValueError(f"Type reference has to be subclass of {StaticEmbedding}.")
        return cls


class AutoStaticDocumentEmbedding(AutoStaticEmbedding):
    @staticmethod
    def from_default_config(repo_id: str, **kwargs: Any) -> FlairDocumentPoolEmbedding:
        embedding = AutoStaticWordEmbedding.from_default_config(repo_id, **kwargs)
        return FlairDocumentPoolEmbedding(embedding)

    @staticmethod
    def from_config(config: StaticModelHubConfig, **kwargs: Any) -> FlairDocumentPoolEmbedding:
        embedding = AutoStaticWordEmbedding.from_config(config, **kwargs)
        return FlairDocumentPoolEmbedding(embedding)
