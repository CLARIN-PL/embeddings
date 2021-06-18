import logging
import os
from typing import Union

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
