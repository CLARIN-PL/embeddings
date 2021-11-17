import abc
from typing import Any, Dict


class LightningPipeline(abc.ABC):

    @abc.abstractmethod
    def run(self) -> Dict[str, Any]:
        pass

    @staticmethod
    def instantiate_kwargs(kwargs: Any) -> Any:
        if kwargs is None:
            kwargs = {}
        return kwargs
