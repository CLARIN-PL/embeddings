import abc
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

import datasets
import pytorch_lightning as pl
from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BatchEncoding

from embeddings.utils.loggers import get_logger

Data = TypeVar("Data")
HuggingFaceDataset = TypeVar("HuggingFaceDataset")

_logger = get_logger(__name__)


class BaseDataModule(abc.ABC, pl.LightningDataModule, Generic[Data]):
    dataset: Data


class HuggingFaceDataModule(BaseDataModule[DatasetDict]):
    LOADER_COLUMNS = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]
    num_labels: int

    def __init__(
        self,
        model_name_or_path: str,
        dataset_name: str,
        target_field: str,
        max_seq_length: Optional[int] = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # ignoring the type to avoid calling to untyped function "__init__" in typed context error
        # caused by pl.LightningDataModule __init__ method not being typed
        super().__init__()  # type: ignore
        self.model_name_or_path = model_name_or_path
        self.dataset_name = dataset_name
        self.target_field = target_field
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.initialized = False
        if load_dataset_kwargs is None:
            self.load_dataset_kwargs = {}
        else:
            self.load_dataset_kwargs = load_dataset_kwargs

    @property
    def output_dim(self) -> Optional[int]:
        if not self.initialized:
            _logger.warning("Datamodule not initialized. Returning None.")
            return None
        else:
            return self.num_labels

    def prepare_data(self) -> None:
        datasets.load_dataset(self.dataset_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.initialized:
            self.initalize()
        if stage in (None, "fit"):
            self.process_data(stage)

    def process_data(self, stage: Optional[str] = None) -> None:
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=[self.target_field],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.LOADER_COLUMNS]
            self.dataset[split].set_format(type="torch", columns=self.columns)

    def train_dataloader(self) -> DataLoader[HuggingFaceDataset]:
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size)

    # Ignoring the type of val_dataloader method from supertype "DataHooks" allowing for None
    # and training without validation dataset.
    def val_dataloader(self) -> Optional[DataLoader[HuggingFaceDataset]]:  # type: ignore
        if "validation" in self.dataset:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        else:
            return None

    def test_dataloader(self) -> DataLoader[HuggingFaceDataset]:
        return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)

    @abc.abstractmethod
    def initalize(self) -> None:
        """Initialize dataset to populate important attributes needed before calling setup."""
        pass

    @abc.abstractmethod
    def convert_to_features(
        self, example_batch: Dict[str, Any], indices: Optional[List[int]] = None
    ) -> BatchEncoding:
        pass


class TextClassificationDataModule(HuggingFaceDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        dataset_name: str,
        text_fields: Union[str, List[str]],
        target_field: str,
        max_seq_length: Optional[int] = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        if isinstance(text_fields, str):
            text_fields = [text_fields]
        if len(text_fields) > 2:
            raise ValueError("Too many fields given in text_fields attribute")
        self.text_fields = text_fields
        super().__init__(
            model_name_or_path=model_name_or_path,
            dataset_name=dataset_name,
            target_field=target_field,
            max_seq_length=max_seq_length,
            train_batch_siz=train_batch_size,
            eval_batch_size=eval_batch_size,
            load_dataset_kwargs=load_dataset_kwargs,
            **kwargs,
        )

    def initalize(self) -> None:
        self.dataset = datasets.load_dataset(self.dataset_name, **self.load_dataset_kwargs)
        self.num_labels = len(set(ex[self.target_field] for ex in self.dataset["train"]))
        self.initialized = True

    def convert_to_features(
        self, example_batch: Dict[str, Any], indices: Optional[List[int]] = None
    ) -> BatchEncoding:
        """Encodes either single sentence or sentence pairs."""
        if len(self.text_fields) == 2:
            texts_or_text_pairs = list(
                zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]])
            )
        elif len(self.text_fields) == 1:
            texts_or_text_pairs = example_batch[self.text_fields[0]]
        else:
            raise ValueError("Inappropriate length of text_fields attribute")

        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            truncation=True,
        )

        features["labels"] = example_batch[self.target_field]

        return features
