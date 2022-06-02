from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, Optional, Tuple

import pandas as pd
import yaml

from embeddings.data.io import T_path
from embeddings.pipeline.pipelines_metadata import Metadata
from embeddings.utils.loggers import LightningLoggingConfig, WandbWrapper
from embeddings.utils.results_persister import ResultsPersister
from embeddings.utils.utils import standardize_name


@dataclass
class HPSResultsPersister(ResultsPersister[Tuple[pd.DataFrame, Metadata]], Generic[Metadata]):
    best_params_path: T_path
    log_path: T_path
    logging_config: LightningLoggingConfig = LightningLoggingConfig()
    logging_hps_summary_name: Optional[str] = None

    def persist(self, result: Tuple[pd.DataFrame, Metadata], **kwargs: Any) -> None:
        log, metadata = result
        log.to_pickle(self.log_path)
        with open(self.best_params_path, "w") as f:
            yaml.dump(data=metadata, stream=f)
        if self.logging_config.use_wandb():
            general_metadata = deepcopy(metadata)
            del general_metadata["config"]
            logger = WandbWrapper()
            assert "embedding_name_or_path" in general_metadata
            logging_hps_summary_name = (
                standardize_name(self.logging_hps_summary_name)
                if self.logging_hps_summary_name
                else self.logging_hps_summary_name
            )
            logger.init_logging(
                name=logging_hps_summary_name,
                project_name=self.logging_config.tracking_project_name,
                entity=self.logging_config.wandb_entity,
                config=general_metadata,
                **self.logging_config.wandb_logger_kwargs,
            )
            logger.log_artifact(
                paths=[self.log_path, self.best_params_path],
                artifact_name="hps_result",
                artifact_type="output",
            )
            logger.finish_logging()
