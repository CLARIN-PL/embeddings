import abc
from typing import Any


class Transformer(abc.ABC):
    model: Any

    def freeze_transformer(self) -> None:
        for param in self.model.base_model.parameters():
            param.requires_grad = False

    def unfreeze_transformer(self, unfreeze_from: int = -1) -> None:
        if unfreeze_from == -1:
            for param in self.model.base_model.parameters():
                param.requires_grad = True
        else:
            requires_grad = False
            for name, param in self.model.base_model.named_parameters():
                if not requires_grad:
                    if name.startswith("encoder.layer"):
                        no_layer = int(name.split(".")[2])
                        if no_layer >= unfreeze_from:
                            requires_grad = True
                param.requires_grad = requires_grad

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass
