import abc
from typing import Any

from transformers import PreTrainedModel


class Transformer(abc.ABC):
    model: PreTrainedModel

    def __init__(self, **kwargs: Any) -> None:
        pass

    def freeze_transformer(self) -> None:
        for param in self.model.base_model.parameters():
            param.requires_grad = False

    def unfreeze_transformer(self, unfreeze_from: int = -1) -> None:
        for name, param in self.model.base_model.named_parameters():
            if name.startswith("encoder.layer"):
                no_layer = int(name.split(".")[2])
                if no_layer >= unfreeze_from:
                    param.requires_grad = True

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass
