from typing import Any, List, Literal

import torch


class DocumentPoolEmbedding:
    SUPPORTED_STRATEGIES: List[str] = ["mean", "cls", "max"]

    def __init__(self, strategy: Literal["mean", "cls"]):
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError("Wrong strategy given as the argument")
        self.strategy = strategy

    def __call__(self, model_output: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.strategy == "mean":
            return model_output.mean(dim=1)
        elif self.strategy == "cls":
            return model_output[:, 0, :]
        elif self.strategy == "max":
            return model_output.max(dim=1).values
