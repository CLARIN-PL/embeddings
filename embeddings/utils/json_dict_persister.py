from typing import Any, Dict

import srsly

from embeddings.utils.results_persister import ResultsPersister


class JsonPersister(ResultsPersister[Dict[str, Any]]):
    def __init__(self, path: str):
        self.path = path

    def persist(self, result: Dict[str, Any], **kwargs: Any) -> None:
        srsly.write_json(self.path, result)
