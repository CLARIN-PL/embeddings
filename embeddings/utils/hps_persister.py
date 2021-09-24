from dataclasses import dataclass
from typing import Any, Generic, Tuple

import pandas as pd
import yaml

from embeddings.data.io import T_path
from embeddings.pipeline.pipelines_metadata import Metadata
from embeddings.utils.results_persister import ResultsPersister


@dataclass
class HPSResultsPersister(ResultsPersister[Tuple[pd.DataFrame, Metadata]], Generic[Metadata]):
    best_params_path: T_path
    log_path: T_path

    def persist(self, result: Tuple[pd.DataFrame, Metadata], **kwargs: Any) -> None:
        result[0].to_pickle(self.log_path)
        with open(self.best_params_path, "w") as f:
            yaml.dump(data=result[1], stream=f)
