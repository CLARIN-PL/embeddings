from typing import Any, Dict

import srsly

from embeddings.utils.results_persister import ResultsPersister


class JsonPersister(ResultsPersister[Dict[str, Any]]):
    def __init__(self, path: str):
        self.path = path

    def persist(self, result: Dict[str, Any], **kwargs: Any) -> None:
        with open(self.path, "w") as json_file:
            json_file.write(srsly.json_dumps(result))
