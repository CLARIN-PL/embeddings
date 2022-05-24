import abc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import wandb
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import LightningLoggerBase
from typing_extensions import Literal

from embeddings.data.io import T_path

DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", logging.INFO)


def get_logger(name: str, log_level: Union[str, int] = DEFAULT_LOG_LEVEL) -> logging.Logger:
    log_level = log_level
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


@dataclass
class LightningLoggingConfig:
    loggers_names: List[Literal["wandb", "csv", "tensorboard"]] = field(default_factory=list)
    tracking_project_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_logger_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "wandb" not in self.loggers_names and (
            self.tracking_project_name or self.wandb_entity or self.wandb_logger_kwargs
        ):
            raise ValueError(
                "`wandb_project` or `wandb_entity` or 'wandb_logger_kwargs' was configured but "
                "use_wand is set to false."
            )

    @classmethod
    def from_flags(
        cls,
        wandb: bool = False,
        csv: bool = False,
        tensorboard: bool = False,
        tracking_project_name: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_logger_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "LightningLoggingConfig":
        loggers_names: list[Literal["wandb", "csv", "tensorboard"]] = []
        if wandb:
            loggers_names.append("wandb")
        if csv:
            loggers_names.append("csv")
        if tensorboard:
            loggers_names.append("tensorboard")

        return cls(
            loggers_names=loggers_names,
            tracking_project_name=tracking_project_name,
            wandb_entity=wandb_entity,
            wandb_logger_kwargs=wandb_logger_kwargs or {},
        )

    def use_wandb(self) -> bool:
        return "wandb" in self.loggers_names

    def use_csv(self) -> bool:
        return "csv" in self.loggers_names

    def use_tensorboard(self) -> bool:
        return "tensorboard" in self.loggers_names

    def get_lightning_loggers(
        self,
        output_path: T_path,
        run_name: Optional[str] = None,
    ) -> List[LightningLoggerBase]:
        """Based on configuration, provides pytorch-lightning loggers' callbacks."""
        output_path = Path(output_path)
        loggers: List[LightningLoggerBase] = []

        if self.use_tensorboard():
            loggers.append(
                pl_loggers.TensorBoardLogger(
                    name=run_name,
                    save_dir=str(output_path.joinpath("tensorboard")),
                )
            )

        if self.use_wandb():
            save_dir = output_path.joinpath("wandb")
            save_dir.mkdir(exist_ok=True)
            loggers.append(
                pl_loggers.WandbLogger(
                    name=run_name,
                    save_dir=str(save_dir),
                    project=self.tracking_project_name,
                    entity=self.wandb_entity,
                    reinit=True,
                    **self.wandb_logger_kwargs
                )
            )

        if self.use_csv():
            loggers.append(
                pl_loggers.CSVLogger(name=run_name, save_dir=str(output_path.joinpath("csv")))
            )

        return loggers


class ExperimentLogger(abc.ABC):
    @abc.abstractmethod
    def log_output(self, output_path: T_path, run_name: Optional[str] = None) -> None:
        pass

    @abc.abstractmethod
    def init_logging(self) -> None:
        pass

    @abc.abstractmethod
    def finish_logging(self) -> None:
        pass

    @abc.abstractmethod
    def log_artifact(self, paths: Iterable[T_path], artifact_name: str, artifact_type: str) -> None:
        pass


class WandbWrapper(ExperimentLogger):
    def log_output(
        self,
        output_path: T_path,
        ignore: Optional[Iterable[str]] = None,
    ) -> None:
        for entry in os.scandir(output_path):
            if not ignore or entry.name not in ignore:
                wandb.save(entry.path, output_path)

    def init_logging(
        self,
        name: Optional[str] = None,
        project_name: Optional[str] = None,
        entity: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        assert wandb.run is None, "The last logging is not finished."
        wandb.init(name=name, project=project_name, entity=entity, **kwargs)

    def finish_logging(self) -> None:
        wandb.finish()

    def log_artifact(self, paths: Iterable[T_path], artifact_name: str, artifact_type: str) -> None:
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
        for path in paths:
            artifact.add_file(path)
        wandb.log_artifact(artifact)
