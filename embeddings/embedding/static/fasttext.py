from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from flair.embeddings import FastTextEmbeddings

from embeddings.embedding.flair_embedding import FlairEmbedding
from embeddings.embedding.static.config import SingleFileConfig, StaticModelHubConfig
from embeddings.embedding.static.word import StaticWordEmbedding


@dataclass
class KGR10FastTextConfig(SingleFileConfig):
    method: Optional[Literal["cbow", "skipgram"]] = None
    dimension: Optional[Literal[100, 300]] = None
    model_name: str = field(default=f"kgr10.plain.{method}.dim{dimension}.neg10.bin", init=False)
    repo_id: str = field(default="clarin-pl/fastText-kgr10", init=False)

    def __post_init__(self) -> None:
        if not self.method:
            self.method = self.default_config["method"]

        if not self.dimension:
            self.dimension = self.default_config["dimension"]

        self.model_name = f"kgr10.plain.{self.method}.dim{self.dimension}.neg10.bin"


class SingleFileEmbedding(StaticWordEmbedding, ABC):
    def __init__(self, config: SingleFileConfig, **load_model_kwargs: Any):
        super().__init__(config.cached_model, **load_model_kwargs)
        self.config = config


class FlairFastTextEmbedding(FlairEmbedding):
    def _get_model(self) -> FastTextEmbeddings:
        return FastTextEmbeddings(self.name, **self.load_model_kwargs)


class KGR10FastTextEmbedding(SingleFileEmbedding, FlairFastTextEmbedding):
    @staticmethod
    def get_config(**kwargs: Any) -> KGR10FastTextConfig:
        return KGR10FastTextConfig(**kwargs)

    @staticmethod
    def from_default_config(
        config: StaticModelHubConfig, **kwargs: Any
    ) -> "KGR10FastTextEmbedding":
        config = KGR10FastTextEmbedding.get_config(**config.default_config)
        return KGR10FastTextEmbedding(config)
