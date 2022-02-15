from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Type, Union

from embeddings.data.io import T_path
from embeddings.embedding import auto_flair
from embeddings.embedding.flair_embedding import FlairDocumentPoolEmbedding
from embeddings.utils.loggers import get_logger

_logger = get_logger(__name__)


class FlairEmbeddingLoader:
    def __init__(self, embedding_name: T_path, model_type_reference: Optional[str] = None):
        self.embedding_name = embedding_name
        self.model_type_reference = model_type_reference

    @abstractmethod
    def get_embedding(self) -> auto_flair.FlairEmbedding:
        """Loads an embedding model from hugging face hub or from file that is stored locally,
        if the model is compatible with Transformers or the current library
        """
        pass


class FlairDocumentPoolEmbeddingLoader(FlairEmbeddingLoader):
    def get_embedding(
        self,
        document_embedding_cls: Union[
            str, Type[auto_flair.DocumentEmbedding]
        ] = FlairDocumentPoolEmbedding,
        **load_model_kwargs
    ) -> auto_flair.FlairEmbedding:
        if not isinstance(self.embedding_name, Path):
            return auto_flair.AutoFlairDocumentPoolEmbedding.from_hub(
                repo_id=self.embedding_name,
                document_embedding_cls=document_embedding_cls,
                **load_model_kwargs or {}
            )

        if not self.model_type_reference:
            _logger.error(
                "For embedding loaded directly from file model_type_reference must be provided!"
            )

        return auto_flair.AutoFlairDocumentPoolEmbedding.from_file(
            file_path=self.embedding_name,
            model_type_reference=self.model_type_reference,
            document_embedding_cls=document_embedding_cls,
            **load_model_kwargs or {}
        )


class FlairDocumentEmbeddingLoader(FlairEmbeddingLoader):
    def get_embedding(self, **load_model_kwargs) -> auto_flair.FlairEmbedding:
        if not isinstance(self.embedding_name, Path):
            return auto_flair.AutoFlairDocumentEmbedding.from_hub(
                repo_id=self.embedding_name, **load_model_kwargs or {}
            )

        if not self.model_type_reference:
            _logger.error(
                "For embedding loaded directly from file model_type_reference must be provided!"
            )

        return auto_flair.AutoFlairDocumentEmbedding.from_file(
            file_path=self.embedding_name,
            model_type_reference=self.model_type_reference,
            **load_model_kwargs or {}
        )


class FlairWordEmbeddingLoader(FlairEmbeddingLoader):
    def get_embedding(self, **load_model_kwargs) -> auto_flair.FlairEmbedding:
        if not isinstance(self.embedding_name, Path):
            return auto_flair.AutoFlairWordEmbedding.from_hub(
                self.embedding_name, **load_model_kwargs or {}
            )

        if not self.model_type_reference:
            _logger.error(
                "For embedding loaded directly from file model_type_reference must be provided!"
            )

        return auto_flair.AutoFlairWordEmbedding.from_file(
            self.embedding_name, self.model_type_reference, **load_model_kwargs or {}
        )
