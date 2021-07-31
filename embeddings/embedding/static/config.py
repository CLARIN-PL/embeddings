from dataclasses import dataclass
from typing import Any

import srsly
from huggingface_hub import cached_download, hf_hub_url


@dataclass
class StaticModelHubConfig:
    repo_id: str

    @property
    def model_type_reference(self) -> str:
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
class SingleFileConfig(StaticModelHubConfig):
    model_name: str

    @property
    def cached_model(self) -> str:
        url: str = self._get_file_hf_hub_url(self.model_name)
        path: str = cached_download(url)
        return path
