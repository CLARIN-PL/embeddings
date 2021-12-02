from collections import ChainMap
from typing import Any, Optional

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from transformers.file_utils import ModelOutput  # type: ignore

from embeddings.model.lightning.sequence_classification import SequenceClassificationModule
from embeddings.model.lightning.sequence_labeling import SequenceLabelingModule
from embeddings.model.lightning.transformer import Transformer


class AutoTransformer(Transformer):
    def __init__(
        self,
        model_name_or_path: str,
        model_cls: Any,
        num_labels: int,
        unfreeze_from: Optional[int] = None,
        **kwargs: Any
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = model_cls.from_pretrained(model_name_or_path, config=self.config)
        self.freeze_transformer()
        if unfreeze_from is not None:
            self.unfreeze_transformer(unfreeze_from=unfreeze_from)

    def forward(self, *args: Any, **kwargs: Any) -> ModelOutput:
        assert (not (args and kwargs)) and (args or kwargs)
        inputs = kwargs if kwargs else args
        if isinstance(inputs, tuple):
            inputs = dict(ChainMap(*inputs))
        outputs = self.model(**inputs)
        return outputs


class AutoTransformerForSequenceClassification(AutoTransformer, SequenceClassificationModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        unfreeze_from: Optional[int] = None,
        **kwargs: Any
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            model_cls=AutoModelForSequenceClassification,
            unfreeze_from=unfreeze_from,
            num_labels=num_labels,
            **kwargs
        )


class AutoTransformerForTokenClassification(AutoTransformer, SequenceLabelingModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        unfreeze_from: Optional[int] = None,
        **kwargs: Any
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            model_cls=AutoModelForTokenClassification,
            unfreeze_from=unfreeze_from,
            num_labels=num_labels,
            **kwargs
        )
