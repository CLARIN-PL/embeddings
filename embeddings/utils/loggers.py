import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from typing_extensions import Literal

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
    loggers_names: list[Literal["wandb", "csv", "tensorboard"]] = field(default_factory=list)
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
