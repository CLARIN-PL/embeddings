from dataclasses import dataclass, field
from typing import Any, Optional

from flair.embeddings import FastTextEmbeddings
from typing_extensions import Literal

from embeddings.embedding.flair_embedding import FlairEmbedding
from embeddings.embedding.static.config import SingleFileConfig, StaticModelHubConfig
from embeddings.embedding.static.embedding import SingleFileEmbedding


@dataclass
class KGR10FastTextConfig(SingleFileConfig):
    method: Optional[Literal["cbow", "skipgram"]] = None
    dimension: Optional[Literal[100, 300]] = None
    model_name: str = field(init=False)
    repo_id: str = field(default="clarin-pl/fastText-kgr10", init=False)

    def __post_init__(self) -> None:
        if not self.method:
            self.method = self.default_config["method"]

        if not self.dimension:
            self.dimension = self.default_config["dimension"]

        self.model_name = f"kgr10.plain.{self.method}.dim{self.dimension}.neg10.bin"

        if not self.file_accessible(self.model_name):
            raise ValueError(
                f"Model for the given configuration is not accessible. Change config."
                f"\nUrl: {self._get_file_hf_hub_url(self.model_name)}"
            )


class FlairFastTextEmbedding(FlairEmbedding):
    def _get_model(self) -> FastTextEmbeddings:
        return FastTextEmbeddings(self.name, **self.load_model_kwargs)


class KGR10FastTextEmbedding(SingleFileEmbedding, FlairFastTextEmbedding):
    @staticmethod
    def from_config(config: StaticModelHubConfig, **kwargs: Any) -> "KGR10FastTextEmbedding":
        if not isinstance(config, KGR10FastTextConfig):
            raise ValueError(f"Wrong config type {type(config)}, expected {KGR10FastTextConfig}.")
        return KGR10FastTextEmbedding(config)

    @staticmethod
    def from_default_config(
        config: StaticModelHubConfig, **kwargs: Any
    ) -> "KGR10FastTextEmbedding":
        config = KGR10FastTextEmbedding.get_config(**config.default_config)
        return KGR10FastTextEmbedding(config)

    @staticmethod
    def get_config(**kwargs: Any) -> KGR10FastTextConfig:
        return KGR10FastTextConfig(**kwargs)
