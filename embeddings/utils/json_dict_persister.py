import dataclasses
import json
from typing import Any, TypeVar

import numpy as np

from embeddings.data.io import T_path
from embeddings.evaluator.evaluation_results import EvaluationResults
from embeddings.utils.results_persister import ResultsPersister

EvaluationResultsType = TypeVar("EvaluationResultsType", bound=EvaluationResults)


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        else:
            return super(CustomJsonEncoder, self).default(obj)


class JsonPersister(ResultsPersister[EvaluationResultsType]):
    def __init__(self, path: T_path):
        self.path = path

    def persist(self, result: EvaluationResultsType, **kwargs: Any) -> None:
        with open(self.path, "w") as f:
            json.dump(result, f, cls=CustomJsonEncoder, indent=2)
