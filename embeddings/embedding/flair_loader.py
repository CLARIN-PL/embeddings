from abc import abstractmethod
from pathlib import Path
from typing import Any, Type, Union

from embeddings.data.io import T_path
from embeddings.embedding.auto_flair import (
    AutoFlairDocumentEmbedding,
    AutoFlairDocumentPoolEmbedding,
    AutoFlairWordEmbedding,
    DocumentEmbedding,
)
from embeddings.embedding.flair_embedding import FlairDocumentPoolEmbedding, FlairEmbedding


class FlairEmbeddingLoader:
    def __init__(self, embedding_name: T_path, model_type_reference: str):
        self.embedding_name = embedding_name
        self.model_type_reference = model_type_reference

    @abstractmethod
    def get_embedding(self) -> FlairEmbedding:
        """Loads an embedding model from hugging face hub or from file that is stored locally,
        if the model is compatible with Transformers or the current library
        """
        pass


class FlairDocumentPoolEmbeddingLoader(FlairEmbeddingLoader):
    def get_embedding(
        self,
        document_embedding_cls: Union[str, Type[DocumentEmbedding]] = FlairDocumentPoolEmbedding,
        **load_model_kwargs: Any
    ) -> FlairEmbedding:
        if not isinstance(self.embedding_name, Path):
            return AutoFlairDocumentPoolEmbedding.from_hub(
                repo_id=self.embedding_name,
                document_embedding_cls=document_embedding_cls,
                **load_model_kwargs or {}
            )

        return AutoFlairDocumentPoolEmbedding.from_file(
            file_path=self.embedding_name,
            model_type_reference=self.model_type_reference,
            document_embedding_cls=document_embedding_cls,
            **load_model_kwargs or {}
        )


class FlairDocumentEmbeddingLoader(FlairEmbeddingLoader):
    def get_embedding(self, **load_model_kwargs: Any) -> FlairEmbedding:
        if not isinstance(self.embedding_name, Path):
            return AutoFlairDocumentEmbedding.from_hub(
                repo_id=self.embedding_name, **load_model_kwargs or {}
            )

        return AutoFlairDocumentEmbedding.from_file(
            file_path=self.embedding_name,
            model_type_reference=self.model_type_reference,
            **load_model_kwargs or {}
        )


class FlairWordEmbeddingLoader(FlairEmbeddingLoader):
    def get_embedding(self, **load_model_kwargs: Any) -> FlairEmbedding:
        if not isinstance(self.embedding_name, Path):
            return AutoFlairWordEmbedding.from_hub(self.embedding_name, **load_model_kwargs or {})

        return AutoFlairWordEmbedding.from_file(
            self.embedding_name, self.model_type_reference, **load_model_kwargs or {}
        )
