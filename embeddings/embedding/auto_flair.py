from abc import ABC, abstractmethod
from typing import Any

from embeddings.embedding.flair_embedding import (
    FlairEmbedding,
    FlairTransformerDocumentEmbedding,
    FlairTransformerWordEmbedding,
)
from embeddings.embedding.static.word import AutoStaticDocumentEmbedding, AutoStaticWordEmbedding


class AutoFlairEmbedding(ABC):
    @staticmethod
    @abstractmethod
    def from_hub(repo_id: str, **kwargs: Any) -> FlairEmbedding:
        pass


class AutoFlairWordEmbedding(AutoFlairEmbedding):
    @staticmethod
    def from_hub(repo_id: str, **kwargs: Any) -> FlairEmbedding:
        """Loads an embedding model from hugging face hub, if the model is compatible with
        Transformers or the current library.

        In case of StaticWordEmbedding, a default config is used during initialisation. If you
        rather want to specify custom config, use the AutoStaticWordEmbedding.from_config function.
        """

        try:
            return FlairTransformerWordEmbedding(repo_id, **kwargs)
        except EnvironmentError:
            return AutoStaticWordEmbedding.from_default_config(repo_id, **kwargs)


class AutoFlairDocumentEmbedding(AutoFlairEmbedding):
    @staticmethod
    def from_hub(repo_id: str, **kwargs: Any) -> FlairEmbedding:
        """Loads an embedding model from hugging face hub, if the model is compatible with
        Transformers or the current library.

        In case of StaticWordEmbedding, a default config is used during initialisation. If you
        rather want to specify custom config, use the AutoFlairDocumentEmbedding.from_config
        function.

        Additionally, in case of StaticWordEmbedding, pooling ("mean" by default) of such embeddings
         is performed to obtain a document's embedding.
        """

        try:
            return FlairTransformerDocumentEmbedding(repo_id, **kwargs)
        except EnvironmentError:
            return AutoStaticDocumentEmbedding.from_default_config(repo_id, **kwargs)
