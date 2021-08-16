import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, cast
from urllib.parse import urlparse

import appdirs
import requests
from numpy import ndarray
from pymagnitude import Magnitude

from embeddings.embedding.embedding import Embedding
from embeddings.utils.loggers import get_logger

_logger = get_logger(__name__)


@dataclass
class MagnitudeConfig:
    model_path: str


class MagnitudeEmbedding(Embedding[List[str], ndarray]):
    def __init__(self, config: MagnitudeConfig) -> None:
        self.model = Magnitude(config.model_path)

    def embed(self, data: List[str]) -> ndarray:
        return cast(ndarray, self.model.query(data))


class AutoMagnitude:
    @staticmethod
    def from_url(url: str) -> MagnitudeEmbedding:
        cache_dir = appdirs.user_cache_dir("embeddings", "magnitude")
        parsed_url = urlparse(url)
        model_file = os.path.join(
            cache_dir,
            f"{hashlib.md5(url.encode('utf-8')).hexdigest()}{os.path.basename(parsed_url.path)}",
        )
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        try:
            _logger.info("Trying to load Magnitude model from cache")
            return MagnitudeEmbedding(MagnitudeConfig(model_file))
        except:
            _logger.info("Fetching model from URL as it is not cached")
            result = requests.get(url, allow_redirects=True)
            open(model_file, "wb+").write(result.content)
            return MagnitudeEmbedding(MagnitudeConfig(model_file))
