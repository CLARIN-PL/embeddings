import abc
from typing import Any, Dict, Generic, List, Optional, Sequence, Type, TypeVar, Union

import datasets
import pytorch_lightning as pl
from datasets import ClassLabel, Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BatchEncoding

from embeddings.utils.loggers import get_logger

Data = TypeVar("Data")
HuggingFaceDataset = Type[Dataset]
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
    DEFAULT_TOKENIZER_KWARGS = {"use_fast": True}
    DEFAULT_BATCH_ENCODING_KWARGS = {
        "padding": True,
        "truncation": True,
    }

    def __init__(
        self,
        tokenizer_name_or_path: str,
        dataset_name: str,
        target_field: str,
        max_seq_length: Optional[int] = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        downsample_train: Optional[float] = None,
        downsample_val: Optional[float] = None,
        downsample_test: Optional[float] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        batch_encoding_kwargs: Optional[Dict[str, Any]] = None,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # ignoring the type to avoid calling to untyped function "__init__" in typed context error
        # caused by pl.LightningDataModule __init__ method not being typed
        super().__init__()  # type: ignore
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.dataset_name = dataset_name
        self.target_field = target_field
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.downsample_train = downsample_train
        self.downsample_val = downsample_val
        self.downsample_test = downsample_test
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path,
            **tokenizer_kwargs if tokenizer_kwargs else self.DEFAULT_TOKENIZER_KWARGS,
        )
        self.batch_encoding_kwargs = (
            batch_encoding_kwargs if batch_encoding_kwargs else self.DEFAULT_BATCH_ENCODING_KWARGS
        )
        self.load_dataset_kwargs = load_dataset_kwargs if load_dataset_kwargs else {}

    def load_dataset(self) -> DatasetDict:
        result = datasets.load_dataset(self.dataset_name, **self.load_dataset_kwargs)
        assert isinstance(result, DatasetDict)
        return result

    def get_num_classes(self) -> int:
        assert isinstance(self.dataset, DatasetDict)
        if not isinstance(self.dataset["train"].features[self.target_field], ClassLabel):
            self.dataset.class_encode_column(self.target_field)
        num_classes = self.dataset["train"].features[self.target_field].num_classes
        assert isinstance(num_classes, int)
        return num_classes

    def downsample_dataset(self) -> None:
        assert isinstance(self.dataset, DatasetDict)
        downsamples = (
            ("train", self.downsample_train),
            ("validation", self.downsample_val),
            ("test", self.downsample_test),
        )
        for column_name, downsample_factor in downsamples:
            if (
                downsample_factor is not None
                and column_name in self.dataset
                and 0 < downsample_factor < 1
            ):
                downsampled_data = self.dataset[column_name].train_test_split(downsample_factor)
                self.dataset[column_name] = downsampled_data["test"]

    def prepare_data(self) -> None:
        datasets.load_dataset(self.dataset_name)
        AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = self.load_dataset()
        self.downsample_dataset()
        self.num_classes = self.get_num_classes()
        self.process_data()

    def process_data(self) -> None:
        columns = [c for c in self.dataset["train"].column_names if c not in self.LOADER_COLUMNS]
        self.dataset = self.dataset.map(
            self.convert_to_features,
            batched=True,
            remove_columns=columns,
        )
        self.dataset.set_format(type="torch")

    def train_dataloader(self) -> DataLoader[HuggingFaceDataset]:
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size)  # type: ignore

    # Ignoring the type of val_dataloader method from supertype "DataHooks" allowing for None
    # and training without validation dataset.
    def val_dataloader(self) -> Optional[DataLoader[HuggingFaceDataset]]:  # type: ignore
        if "validation" in self.dataset:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)  # type: ignore
        else:
            return None

    def test_dataloader(self) -> DataLoader[HuggingFaceDataset]:
        return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)  # type: ignore

    @abc.abstractmethod
    def convert_to_features(
        self, example_batch: Dict[str, Any], indices: Optional[List[int]] = None
    ) -> BatchEncoding:
        pass


class TextClassificationDataModule(HuggingFaceDataModule):
    def __init__(
        self,
        tokenizer_name_or_path: str,
        dataset_name: str,
        text_fields: Union[str, Sequence[str]],
        target_field: str,
        max_seq_length: Optional[int] = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        batch_encoding_kwargs: Optional[Dict[str, Any]] = None,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        if isinstance(text_fields, str):
            text_fields = [text_fields]
        if len(text_fields) > 2:
            raise ValueError("Too many fields given in text_fields attribute")
        self.text_fields = text_fields
        super().__init__(
            tokenizer_name_or_path=tokenizer_name_or_path,
            dataset_name=dataset_name,
            target_field=target_field,
            max_seq_length=max_seq_length,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            tokenizer_kwargs=tokenizer_kwargs,
            batch_encoding_kwargs=batch_encoding_kwargs,
            load_dataset_kwargs=load_dataset_kwargs,
            **kwargs,
        )

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

        features = self.tokenizer(
            texts_or_text_pairs,
            max_length=self.max_seq_length,
            **self.batch_encoding_kwargs,
        )

        features["labels"] = example_batch[self.target_field]

        return features
