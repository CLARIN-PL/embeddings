import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, cast
from urllib.parse import urlparse

import appdirs
import requests
from flair.data import Sentence
from numpy import typing as nptyping
from pymagnitude import Magnitude

from embeddings.embedding.embedding import Embedding
from embeddings.utils.loggers import get_logger

_logger = get_logger(__name__)


@dataclass
class MagnitudeConfig:
    model_path: str


class MagnitudeEmbedding(Embedding[List[str], nptyping.NDArray[Any]]):
    def __init__(self, config: MagnitudeConfig) -> None:
        self.model = Magnitude(config.model_path)

    def embed(self, data: List[str]) -> nptyping.NDArray[Any]:
        return cast(nptyping.NDArray[Any], self.model.query(data))

    def to_flair(self) -> "MagnitudeFlairConnector":
        return MagnitudeFlairConnector(self)


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


class MagnitudeFlairConnector(Embedding[List[Sentence], List[Sentence]]):
    def __init__(self, to_wrap: MagnitudeEmbedding) -> None:
        self.to_wrap = to_wrap

    def embed(self, data: List[Sentence]) -> List[Sentence]:
        sentences_plain_text = [sentence.to_plain_string() for sentence in data]

        embeddings = super().embed(sentences_plain_text)
        for sentence, embedding in zip(data, embeddings):
            sentence.set_embedding(self.__class__.__name__, embedding)

        return data
