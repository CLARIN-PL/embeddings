from dataclasses import dataclass
from typing import Any, Dict
from urllib.error import HTTPError
from urllib.request import urlopen

import requests
import srsly
from huggingface_hub import cached_download, hf_hub_url

from embeddings.utils.loggers import get_logger

_logger = get_logger(__name__)


@dataclass
class StaticModelHubConfig:
    repo_id: str

    @property
    def model_type_reference(self) -> str:
        reference = self._load_hub_json("module.json")["type"]
        if isinstance(reference, str):
            return reference
        else:
            raise ValueError(f"Wrong format of import reference {reference}.")

    @property
    def default_config(self) -> Dict[str, Any]:
        config = self._load_hub_json("default_config.json")
        if isinstance(config, dict):
            return config
        else:
            raise ValueError(f"Wrong format of default config {config}.")

    def _load_hub_json(self, filename: str) -> Any:
        url = self._get_file_hf_hub_url(filename)
        try:
            path = cached_download(url)
        except requests.HTTPError:
            raise EnvironmentError(
                "Repository not found or wrong format of a given model (module.json not found)."
            )
        return srsly.read_json(path)

    def _get_file_hf_hub_url(self, filename: str) -> str:
        url: str = hf_hub_url(self.repo_id, filename=filename)
        return url

    def file_accessible(self, filename: str) -> bool:
        try:
            result: bool = urlopen(self._get_file_hf_hub_url(filename)).getcode() == 200
            return result
        except HTTPError:
            return False


@dataclass
class SingleFileConfig(StaticModelHubConfig):
    model_name: str

    @property
    def cached_model(self) -> str:
        url: str = self._get_file_hf_hub_url(self.model_name)
        path: str = cached_download(url)
        return path


@dataclass
class GensimFileConfig(SingleFileConfig):
    model_name: str

    @property
    def cached_model(self) -> str:
        url: str = self._get_file_hf_hub_url(self.model_name)
        path: str = cached_download(url)

        npy_vectors_url: str = self._get_file_hf_hub_url(f"{self.model_name}.vectors.npy")

        try:
            cached_download(npy_vectors_url, force_filename=f"{path}.vectors.npy")
        except requests.HTTPError:
            _logger.info(f"{self.model_name}.vectors.npy not found, skipping it.")

        return path
