import json
from typing import Any, Dict

import numpy as np

from embeddings.data.io import T_path
from embeddings.utils.results_persister import ResultsPersister


class JsonEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


class JsonPersister(ResultsPersister[Dict[str, Any]]):
    def __init__(self, path: T_path):
        self.path = path

    def persist(self, result: Dict[str, Any], **kwargs: Any) -> None:
        with open(self.path, "w") as f:
            json.dump(result, f, cls=JsonEncoder, indent=2)
