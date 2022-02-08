from typing import Callable

import numpy as np
import pytorch_lightning as pl
import torch


class BestEpochCallback(pl.callbacks.Callback):
    mode_dict = {"min": torch.lt, "max": torch.gt}
    order_dict = {"min": "<", "max": ">"}

    def __init__(self, monitor="val/Loss", mode="min", **kwargs):
        super().__init__(**kwargs)
        assert mode in self.order_dict

        self.monitor = monitor
        self.mode = mode
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.mode == "min" else -torch_inf
        self.best_epoch = 0

    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._update_best_epoch(trainer)

    def _update_best_epoch(self, trainer: pl.Trainer) -> None:
        logs = trainer.callback_metrics
        current = logs[self.monitor].squeeze()

        if self.monitor_op(current, self.best_score.to(current.device)):
            self.best_score = current
            self.best_epoch = trainer.current_epoch
