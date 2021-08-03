from typing import Any, Optional, Union

from embeddings.embedding.flair_embedding import FlairEmbedding, FlairTransformerWordEmbedding
from embeddings.embedding.static.config import StaticModelHubConfig
from embeddings.embedding.static.word import AutoStaticWordEmbedding


class AutoFlairWordEmbedding:
    @staticmethod
    def from_hub(
        repo_id: Optional[str] = None,
        config: Optional[Union[StaticModelHubConfig]] = None,
        **kwargs: Any
    ) -> FlairEmbedding:
        if repo_id and not config:
            try:
                return FlairTransformerWordEmbedding(repo_id, **kwargs)
            except EnvironmentError:
                return AutoStaticWordEmbedding.from_hub(repo_id, config)
        elif not repo_id and config:
            return AutoStaticWordEmbedding.from_hub(repo_id, config)
        else:
            raise ValueError("You should pass repo_id or config, not both.")
