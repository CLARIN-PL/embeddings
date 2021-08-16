from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from embeddings.embedding.static.config import SingleFileConfig, StaticModelHubConfig
from embeddings.embedding.static.embedding import SingleFileEmbedding, StandardStaticWordEmbedding


@dataclass
class KGR10Word2VecConfig(SingleFileConfig):
    method: Optional[Literal["cbow", "skipgram"]] = None
    hs: Optional[bool] = None
    mwe: Optional[bool] = None
    model_name: str = field(init=False)
    repo_id: str = field(default="clarin-pl/word2vec-kgr10", init=False)

    def __post_init__(self) -> None:
        if not self.method:
            self.method = self.default_config["method"]

        if not self.hs:
            self.hs = self.default_config["hs"]

        if not self.mwe:
            self.mwe = self.default_config["mwe"]

        sampling = "hs" if self.hs else "ns"
        ngrams = "mwe" if self.mwe else "plain"

        self.model_name = f"{self.method}.v300.m8.{sampling}.{ngrams}.w2v.gensim"


class KGR10Word2VecEmbedding(SingleFileEmbedding, StandardStaticWordEmbedding):
    @staticmethod
    def from_config(config: StaticModelHubConfig, **kwargs: Any) -> "KGR10Word2VecEmbedding":
        if not isinstance(config, KGR10Word2VecConfig):
            raise ValueError(f"Wrong config type {type(config)}, expected {KGR10Word2VecConfig}.")
        return KGR10Word2VecEmbedding(config)

    @staticmethod
    def from_default_config(
        config: StaticModelHubConfig, **kwargs: Any
    ) -> "KGR10Word2VecEmbedding":
        config = KGR10Word2VecEmbedding.get_config(**config.default_config)
        return KGR10Word2VecEmbedding(config)

    @staticmethod
    def get_config(**kwargs: Any) -> KGR10Word2VecConfig:
        return KGR10Word2VecConfig(**kwargs)
