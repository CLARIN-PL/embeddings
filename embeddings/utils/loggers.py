import abc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import wandb
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers.wandb import WandbLogger
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
    output_path: Union[Path, str] = "."
    loggers_names: List[Literal["wandb", "csv", "tensorboard"]] = field(default_factory=list)
    tracking_project_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_id : Optional[str] = None
    wandb_logger_kwargs: Dict[str, Any] = field(default_factory=dict)
    loggers: Optional[Dict[str, pl_loggers.Logger]] = field(init=False, default=None)

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
        run_name: Optional[str] = None,
    ) -> List[pl_loggers.Logger]:
        """Based on configuration, provides pytorch-lightning loggers' callbacks."""
        if not self.loggers:
            self.output_path = Path(self.output_path)
            self.loggers = {}

            if self.use_tensorboard():
                self.loggers["tensorboard"] = pl_loggers.TensorBoardLogger(
                    name=run_name,
                    save_dir=str(self.output_path / "tensorboard"),
                )

            if self.use_wandb():
                if not self.tracking_project_name:
                    raise ValueError(
                        "Tracking project name is not passed. Pass tracking_project_name argument!"
                    )
                save_dir = self.output_path / "wandb"
                save_dir.mkdir(exist_ok=True, parents=True)

                self.loggers["wandb"] = pl_loggers.wandb.WandbLogger(
                    name=run_name,
                    save_dir=str(save_dir),
                    project=self.tracking_project_name,
                    entity=self.wandb_entity,
                    id=self.wandb_run_id,
                    **self.wandb_logger_kwargs
                )

            if self.use_csv():
                self.loggers["csv"] = pl_loggers.CSVLogger(
                    name=run_name if run_name else "",
                    save_dir=self.output_path / "csv",
                )

        return list(self.loggers.values())


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


class LightningWandbWrapper:
    def __init__(self, logging_config: LightningLoggingConfig) -> None:
        assert logging_config.use_wandb()
        assert isinstance(logging_config.loggers, dict)
        assert "wandb" in logging_config.loggers
        assert isinstance(logging_config.loggers["wandb"], WandbLogger)
        self.wandb_logger: WandbLogger = logging_config.loggers["wandb"]

    def log_output(
        self,
        output_path: T_path,
        ignore: Optional[Iterable[str]] = None,
    ) -> None:
        for entry in os.scandir(output_path):
            if not ignore or entry.name not in ignore:
                self.wandb_logger.experiment.save(entry.path, output_path)

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        self.wandb_logger.log_metrics(metrics)

    def finish_logging(self) -> None:
        self.wandb_logger.experiment.finish()

    def log_artifact(self, paths: Iterable[T_path], artifact_name: str, artifact_type: str) -> None:
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
        for path in paths:
            artifact.add_file(path)
        self.wandb_logger.experiment.log_artifact(artifact)
