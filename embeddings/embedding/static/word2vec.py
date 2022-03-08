from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from typing_extensions import Literal

from embeddings.embedding.static.config import (
    GensimFileConfig,
    GensimLocalFileConfig,
    StaticModelHubConfig,
)
from embeddings.embedding.static.embedding import (
    LocalFileStaticEmbedding,
    SingleFileEmbedding,
    StandardStaticWordEmbedding,
    StandardStaticWordEmbeddingPL,
)


@dataclass
class KGR10Word2VecConfig(GensimFileConfig):
    method: Optional[Literal["cbow", "skipgram"]] = None
    dimension: Optional[Literal[100, 300]] = None
    hs: Optional[bool] = None
    mwe: Optional[bool] = None
    model_name: str = field(init=False)
    repo_id: str = field(default="clarin-pl/word2vec-kgr10", init=False)

    def __post_init__(self) -> None:
        if not self.method:
            self.method = self.default_config["method"]

        if not self.dimension:
            self.dimension = self.default_config["dimension"]

        if self.hs is None:
            self.hs = self.default_config["hs"]

        if self.mwe is None:
            self.mwe = self.default_config["mwe"]

        sampling = "hs" if self.hs else "ns"
        ngrams = "mwe" if self.mwe else "plain"

        self.model_name = f"{self.method}.v{self.dimension}.m8.{sampling}.{ngrams}.w2v.gensim"

        if not self.file_accessible(self.model_name):
            raise ValueError(
                f"Model for the given configuration is not accessible. Change config."
                f"\nUrl: {self._get_file_hf_hub_url(self.model_name)}"
            )


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


@dataclass
class IPIPANWord2VecConfig(GensimLocalFileConfig):
    corpus: str = field(init=False)
    type: str = field(init=False)
    subtype: str = field(init=False)
    dimension: int = field(init=False)
    method: str = field(init=False)
    algorithm: str = field(init=False)
    model_name: str = field(init=False)
    reduced_vocabulary: bool = False
    model_type_reference: str = "embeddings.embedding.static.word2vec.IPIPANWord2VecEmbedding"

    def __post_init__(self) -> None:
        if "-SLASH-" in self.model_file_path.name:
            self.model_name = self.model_file_path.name.split("-SLASH-")[-1].split(".")[0]
        else:
            self.model_name = self.model_file_path.name.split(".")[0]

        metadata = self.model_name.split("-")
        # IPIPan model are named according to convention that allow to get metadata about embedding from its name.
        # Embedding name contain between 6 and 8 elements after splitting it with "-"
        assert len(metadata) in [
            6,
            7,
            8,
        ], "Model filename is not consistent with IPIPAN rules."

        self.corpus = metadata[0]
        self.type = metadata[1]
        self.subtype = metadata[2]
        self.dimension = int(metadata[3])
        self.method = metadata[4]
        self.algorithm = metadata[5]
        if len(metadata) > 6:
            self.reduced_vocabulary = True


class IPIPANWord2VecEmbedding(LocalFileStaticEmbedding, StandardStaticWordEmbeddingPL):
    @staticmethod
    def from_file(file_path: Path, **kwargs: Any) -> "IPIPANWord2VecEmbedding":
        return IPIPANWord2VecEmbedding(str(file_path))

    @staticmethod
    def get_config(**kwargs: Any) -> IPIPANWord2VecConfig:
        return IPIPANWord2VecConfig(**kwargs)

    @staticmethod
    def create_config(file_path: Path, **kwargs: Any) -> IPIPANWord2VecConfig:
        return IPIPANWord2VecConfig(file_path)
