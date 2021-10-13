from abc import ABC, abstractmethod
from typing import Any, Type

from embeddings.embedding import flair_embedding
from embeddings.embedding.flair_embedding import (
    ALLOWED_FLAIR_DOCUMENT_POOLING_CLASSES,
    FlairEmbedding,
    FlairTransformerDocumentEmbedding,
    FlairTransformerWordEmbedding,
)
from embeddings.embedding.static.embedding import (
    AutoStaticDocumentEmbedding,
    AutoStaticWordEmbedding,
)
from embeddings.utils.loggers import get_logger

_logger = get_logger(__name__)


class AutoFlairEmbedding(ABC):
    @staticmethod
    @abstractmethod
    def from_hub(repo_id: str, *args: Any, **kwargs: Any) -> FlairEmbedding:
        """Loads an embedding model from hugging face hub, if the model is compatible with
        Transformers or the current library.

        In case of StaticWordEmbedding, a default config is used during initialisation. If you
        rather want to specify custom config, use the AutoStaticWordEmbedding.from_config function.
        """
        pass

    @staticmethod
    def _log_info_static(repo_id: str) -> None:
        _logger.info(
            f"{repo_id} not compatible with Transformers, trying to initialise as "
            f"static embedding."
        )


class AutoFlairWordEmbedding(AutoFlairEmbedding):
    @staticmethod
    def from_hub(repo_id: str, *args: Any, **kwargs: Any) -> FlairEmbedding:
        try:
            return FlairTransformerWordEmbedding(repo_id, **kwargs)
        except EnvironmentError:
            AutoFlairWordEmbedding._log_info_static(repo_id)
            return AutoStaticWordEmbedding.from_default_config(repo_id, **kwargs)


class AutoFlairDocumentEmbedding(AutoFlairEmbedding):
    @staticmethod
    def from_hub(repo_id: str, *args: Any, **kwargs: Any) -> FlairEmbedding:
        """In case of StaticWordEmbedding, mean pooling on such embeddings is performed to obtain a
        document's embedding.
        """
        try:
            return FlairTransformerDocumentEmbedding(repo_id, **kwargs)
        except EnvironmentError:
            AutoFlairDocumentEmbedding._log_info_static(repo_id)
            return AutoStaticDocumentEmbedding.from_default_config(repo_id, **kwargs)


class AutoFlairDocumentPoolEmbedding(AutoFlairEmbedding):
    @staticmethod
    def from_hub(
        repo_id: str,
        document_pooling: str = "FlairDocumentPoolEmbedding",
        *args: Any,
        **kwargs: Any,
    ) -> FlairEmbedding:
        """AutoFlairDocumentEmbedding that allows to specify pooling for the word embeddings.
        Available document pooling classes:
            -'FlairDocumentPoolEmbedding'
            -'FlairTransformerDocumentEmbedding'
            -'FlairDocumentRNNEmbeddings'
            -'FlairDocumentCNNEmbeddings'
        """
        if document_pooling not in ALLOWED_FLAIR_DOCUMENT_POOLING_CLASSES:
            raise ValueError(f"{document_pooling} not recognized as valid document pooling class.")

        document_pooling_cls: Type[FlairEmbedding] = getattr(flair_embedding, document_pooling)

        if document_pooling_cls == FlairTransformerDocumentEmbedding:
            document_embedding = document_pooling_cls(name=repo_id, **kwargs)
        else:
            document_embedding = document_pooling_cls(
                word_embedding=AutoFlairWordEmbedding.from_hub(repo_id=repo_id), **kwargs
            )

        return document_embedding
