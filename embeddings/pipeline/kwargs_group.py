from dataclasses import dataclass
from typing import Union, Optional, List

from pytorch_lightning.accelerators import Accelerator


@dataclass
class KwargsGroup:
    def parse_fields_to_kwargs(self, kwargs_group_name: str, kwargs_group_fields: dict) -> None:
        setattr(self, kwargs_group_name, {})
        for field_name, field_property in kwargs_group_fields.items():
            if getattr(self, field_name):
                getattr(self, kwargs_group_name)[field_name] = getattr(self, field_name)


@dataclass
class LightningTaskTrainKwargs(KwargsGroup):
    devices: Optional[Union[List[int], str, int]] = None
    max_epochs: Optional[int] = None
    accelerator: Optional[Union[str, Accelerator]] = None


"""
@dataclass
class BatchEncodingKwargs(KwargsGroup):
    batch_padding: Optional[bool] = None
    truncation: Optional[bool] = None
    is_split_into_words: Optional[bool] = None
    return_offsets_mapping: Optional[bool] = None
"""


@dataclass
class DatamoduleKwargs(KwargsGroup):
    max_seq_length: Optional[int] = None
    mini_batch_size: Optional[int] = None


"""
@dataclass
class TokenizerKwargs(KwargsGroup):
    use_fast: Optional[bool] = None
    bos_token: Optional[str] = None
    eos_token: Optional[str] = None
    unk_token: Optional[str] = None
    sep_token: Optional[str] = None
    pad_token: Optional[str] = None
    cls_token: Optional[str] = None
    mask_token: Optional[str] = None
    additional_special_tokens: Optional[str, List[str]] = None
"""


@dataclass
class TaskModelKwargs(KwargsGroup):
    use_scheduler: Optional[bool] = None
    optimizer: Optional[str] = None
    warmup_steps: Optional[int] = None
    learning_rate: Optional[float] = None
    adam_epsilon: Optional[float] = None
    weight_decay: Optional[float] = None
    finetune_last_n_layers: Optional[int] = None
    classifier_dropout: Optional[float] = None


"""
@dataclass
class LoadDatasetKwargs(KwargsGroup):
    pass


@dataclass
class ModelConfigKwargs(KwargsGroup):
    _from_auto: bool = True


@dataclass
class EarlyStoppingKwargs(KwargsGroup):
    monitor: str
    min_delta: float = 0.0
    patience: int = 3
    verbose: bool = False
    mode: str = "min"
    strict: bool = True
    check_finite: bool = True
    stopping_threshold: Optional[float] = None
    divergence_threshold: Optional[float] = None
    check_on_train_epoch_end: Optional[bool] = None
"""
