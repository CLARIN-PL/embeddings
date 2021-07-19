from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import srsly
from flair.embeddings import FastTextEmbeddings
from huggingface_hub import cached_download, hf_hub_url

from embeddings.embedding.flair_embedding import FlairEmbedding


@dataclass
class StaticModelHubConfig:
    repo_id: str

    @property
    def type_reference(self) -> str:
        if isinstance(reference := self._load_hub_json("module.json")["type"], str):
            return reference
        else:
            raise ValueError(f"Wrong format of import reference {reference}.")

    @property
    def default_config(self) -> dict[str, Any]:
        if isinstance(config := self._load_hub_json("default_config.json"), dict):
            return config
        else:
            raise ValueError(f"Wrong format of default config {config}.")

    def _load_hub_json(self, filename: str) -> Any:
        url = self._get_file_hf_hub_url(filename)
        path = cached_download(url)
        return srsly.read_json(path)

    def _get_file_hf_hub_url(self, filename: str) -> str:
        url: str = hf_hub_url(self.repo_id, filename=filename)
        return url


@dataclass
class KGR10FastTextConfig(StaticModelHubConfig):
    method: Optional[Literal["cbow", "skipgram"]] = None
    dimension: Optional[Literal[100, 300]] = None
    repo_id: str = field(default="clarin-pl/fastText-kgr10", init=False)

    def __post_init__(self) -> None:
        if not self.method:
            self.method = self.default_config["method"]

        if not self.dimension:
            self.dimension = self.default_config["dimension"]

    @property
    def model_name(self) -> str:
        return f"kgr10.plain.{self.method}.dim{self.dimension}.neg10.bin"

    @property
    def cached_model(self) -> str:
        url: str = self._get_file_hf_hub_url(self.model_name)
        return url


class FlairFastTextEmbedding(FlairEmbedding):
    def _get_model(self) -> FastTextEmbeddings:
        return FastTextEmbeddings(self.name, **self.load_model_kwargs)


class KGR10FastTextEmbedding(FlairFastTextEmbedding):
    def __init__(self, config: KGR10FastTextConfig, **load_model_kwargs: Any):
        super().__init__(config.cached_model, **load_model_kwargs)

    @staticmethod
    def get_config(config: dict[str, Any]) -> KGR10FastTextConfig:
        return KGR10FastTextConfig(**config)

    @staticmethod
    def from_config(config: StaticModelHubConfig) -> "KGR10FastTextEmbedding":
        if isinstance(config, StaticModelHubConfig):
            config = KGR10FastTextEmbedding.get_config(config.default_config)
        return KGR10FastTextEmbedding(config)
