from typing import Any, Callable, Dict, Final, Optional

import numpy as np
import pytorch_lightning as pl
import torch


class BestEpochCallback(pl.callbacks.Callback):
    MODE_DICT: Final = {
        "min": lambda x, y: (x < y).item(),
        "max": lambda x, y: (x > y).item(),
    }

    def __init__(self, monitor: str = "val/Loss", mode: str = "min", **kwargs: Dict[str, Any]):
        super().__init__(**kwargs)
        assert mode in BestEpochCallback.MODE_DICT

        self.monitor = monitor
        self.mode = mode
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.mode == "min" else -torch_inf
        self.best_epoch: Optional[int] = None

    @property
    def monitor_op(self) -> Callable[[torch.Tensor, torch.Tensor], bool]:
        return BestEpochCallback.MODE_DICT[self.mode]

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._update_best_epoch(trainer)

    def _update_best_epoch(self, trainer: pl.Trainer) -> None:
        logs = trainer.callback_metrics
        if self.monitor not in logs:
            return

        current = logs[self.monitor].squeeze()

        if self.monitor_op(current, self.best_score.to(current.device)):
            self.best_score = current
            self.best_epoch = trainer.current_epoch
