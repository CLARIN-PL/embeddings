from typing import Any

from embeddings.embedding.flair_embedding import FlairEmbedding, FlairTransformerWordEmbedding
from embeddings.embedding.static.word import AutoStaticWordEmbedding


class AutoFlairWordEmbedding:
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
