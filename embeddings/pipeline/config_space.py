from abc import ABC, abstractmethod


class ConfigSpace(ABC):

    @abstractmethod
    def parse_params(self):
        pass


class LightningConfigSpace(ConfigSpace):
    batch_padding": True,
    "truncation": True,
    "is_split_into_words": True,
    "return_offsets_mapping": True,


    def parse_params(self):
        datamodule_kwargs = None
        tokenizer_kwargs = None
        batch_encoding_kwargs = None
        load_dataset_kwargs = None
        task_model_kwargs = None
        task_train_kwargs = None
        model_config_kwargs = None
        early_stopping_kwargs = None
