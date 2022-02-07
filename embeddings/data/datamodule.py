import abc
from os.path import exists, isdir
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Type, TypeVar, Union

import datasets
import pytorch_lightning as pl
from datasets import ClassLabel, Dataset, DatasetDict
from datasets import Sequence as HFSequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BatchEncoding

from embeddings.data import dataset as embeddings_dataset
from embeddings.data.data_collator import CustomDataCollatorForTokenClassification
from embeddings.data.data_loader import HuggingFaceDataLoader, HuggingFaceLocalDataLoader
from embeddings.data.dataset import LightingDataLoaders, LightingDataModuleSubset
from embeddings.data.io import T_path
from embeddings.utils.loggers import get_logger
from embeddings.utils.utils import initialize_kwargs

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

    def __init__(
        self,
        dataset_name_or_path: T_path,
        tokenizer_name_or_path: T_path,
        target_field: str,
        max_seq_length: Optional[int] = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        processing_batch_size: Optional[int] = None,
        downsample_train: Optional[float] = None,
        downsample_val: Optional[float] = None,
        downsample_test: Optional[float] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        seed: int = 441,
        **kwargs: Any,
    ) -> None:
        # ignoring the type to avoid calling to untyped function "__init__" in typed context error
        # caused by pl.LightningDataModule __init__ method not being typed
        super().__init__()  # type: ignore
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.dataset_name_or_path = dataset_name_or_path
        self.target_field = target_field
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.processing_batch_size = processing_batch_size
        self.downsample_train = downsample_train
        self.downsample_val = downsample_val
        self.downsample_test = downsample_test
        self.tokenizer_kwargs = initialize_kwargs(self.DEFAULT_TOKENIZER_KWARGS, tokenizer_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path,
            **self.tokenizer_kwargs,
        )
        self.load_dataset_kwargs = load_dataset_kwargs if load_dataset_kwargs else {}
        self.seed = seed

    @abc.abstractmethod
    def prepare_labels(self) -> None:
        pass

    @abc.abstractmethod
    def _class_encode_column(self, column_name: str) -> None:
        pass

    @abc.abstractmethod
    def convert_to_features(
        self, example_batch: Dict[str, Any], indices: Optional[List[int]] = None
    ) -> BatchEncoding:
        pass

    def prepare_data(self) -> None:
        self.load_dataset(preparation_step=True)
        AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = self.load_dataset()
        self.downsample_dataset()
        self.prepare_labels()
        self.process_data()

    def load_dataset(self, preparation_step: bool = False) -> DatasetDict:
        loader: Union[HuggingFaceDataLoader, HuggingFaceLocalDataLoader] = HuggingFaceDataLoader()
        if exists(self.dataset_name_or_path):
            if not isdir(self.dataset_name_or_path):
                raise NotImplementedError(
                    "Reading from file is currently not supported. "
                    "Pass dataset directory or HuggingFace repository name"
                )

            if preparation_step:
                return datasets.DatasetDict()
            dataset = embeddings_dataset.HuggingFaceDataset(str(self.dataset_name_or_path))
            loader = HuggingFaceLocalDataLoader()
        else:
            dataset = embeddings_dataset.HuggingFaceDataset(
                str(self.dataset_name_or_path), **self.load_dataset_kwargs
            )
        return loader.load(dataset)

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
                downsampled_data = self.dataset[column_name].train_test_split(
                    downsample_factor, seed=self.seed
                )
                self.dataset[column_name] = downsampled_data["test"]

    def process_data(self) -> None:
        columns = [c for c in self.dataset["train"].column_names if c not in self.LOADER_COLUMNS]
        self.dataset = self.dataset.map(
            self.convert_to_features,
            batched=True,
            batch_size=self.processing_batch_size,
            remove_columns=columns,
        )
        self._class_encode_column("labels")
        self.dataset.set_format(type="torch")

    def train_dataloader(self) -> DataLoader[HuggingFaceDataset]:
        return DataLoader(
            dataset=self.dataset["train"],  # type: ignore
            batch_size=self.train_batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    # Ignoring the type of val_dataloader method from supertype "DataHooks" allowing for None
    # and training without validation dataset.
    def val_dataloader(self) -> Optional[DataLoader[HuggingFaceDataset]]:  # type: ignore
        if "validation" in self.dataset:
            return DataLoader(
                dataset=self.dataset["validation"],  # type: ignore
                batch_size=self.eval_batch_size,
                collate_fn=self.collate_fn,
                shuffle=False,
            )
        else:
            return None

    def test_dataloader(self) -> DataLoader[HuggingFaceDataset]:
        return DataLoader(
            dataset=self.dataset["test"],  # type: ignore
            batch_size=self.eval_batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def get_subset(
        self, subset: Union[str, LightingDataModuleSubset]
    ) -> Union[LightingDataLoaders, None]:
        if subset == "train":
            return self.train_dataloader()
        elif subset == "dev":
            return self.val_dataloader()
        elif subset == "test":
            return self.test_dataloader()
        elif subset == "predict":
            raise NotImplementedError("Predict subset not available in HuggingFaceDataModule")
        else:
            raise ValueError("Unrecognized LightingDataModuleSubset")

    @property
    def collate_fn(self) -> Optional[Callable[[Any], Any]]:
        return None


class TextClassificationDataModule(HuggingFaceDataModule):
    DEFAULT_BATCH_ENCODING_KWARGS = {
        "padding": True,
        "truncation": True,
    }

    def __init__(
        self,
        dataset_name_or_path: T_path,
        tokenizer_name_or_path: T_path,
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
        self.batch_encoding_kwargs = initialize_kwargs(
            self.DEFAULT_BATCH_ENCODING_KWARGS, batch_encoding_kwargs
        )
        super().__init__(
            dataset_name_or_path=dataset_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            target_field=target_field,
            max_seq_length=max_seq_length,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            tokenizer_kwargs=tokenizer_kwargs,
            load_dataset_kwargs=load_dataset_kwargs,
            **kwargs,
        )

    def prepare_labels(self) -> None:
        assert isinstance(self.dataset, DatasetDict)
        if not isinstance(self.dataset["train"].features[self.target_field], ClassLabel):
            self._class_encode_column(self.target_field)
        self.num_classes = self.dataset["train"].features[self.target_field].num_classes
        self.target_names = self.dataset["train"].features[self.target_field].names

    def _class_encode_column(self, column_name: str) -> None:
        self.dataset = self.dataset.class_encode_column(column_name)

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


class SequenceLabelingDataModule(HuggingFaceDataModule):
    IGNORE_INDEX = -100
    DEFAULT_BATCH_ENCODING_KWARGS = {
        "padding": True,
        "truncation": True,
        "is_split_into_words": True,
        "return_offsets_mapping": True,
    }

    def __init__(
        self,
        dataset_name_or_path: T_path,
        tokenizer_name_or_path: T_path,
        text_field: str,
        target_field: str,
        max_seq_length: Optional[int] = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        label_all_tokens: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        batch_encoding_kwargs: Optional[Dict[str, Any]] = None,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self.text_field = text_field
        self.label_all_tokens = label_all_tokens
        self.batch_encoding_kwargs = initialize_kwargs(
            self.DEFAULT_BATCH_ENCODING_KWARGS, batch_encoding_kwargs
        )
        super().__init__(
            dataset_name_or_path=dataset_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            target_field=target_field,
            max_seq_length=max_seq_length,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            tokenizer_kwargs=tokenizer_kwargs,
            load_dataset_kwargs=load_dataset_kwargs,
            **kwargs,
        )

    def prepare_labels(self) -> None:
        if not isinstance(self.dataset["train"].features[self.target_field].feature, ClassLabel):
            raise TypeError(
                "Target field has inappropiate type; datasets.Sequence(datasets.features.ClassLabel(...)) is required for Sequence Labeling task"
            )
        else:
            self.num_classes = self.dataset["train"].features[self.target_field].feature.num_classes
            self.target_names = self.dataset["train"].features[self.target_field].feature.names

    def convert_to_features(
        self, example_batch: Dict[str, Any], indices: Optional[List[int]] = None
    ) -> BatchEncoding:
        texts = example_batch[self.text_field]
        features = self.tokenizer.batch_encode_plus(
            texts, max_length=self.max_seq_length, **self.batch_encoding_kwargs
        )
        labels = self.encode_tags(labels=example_batch[self.target_field], encodings=features)
        features["labels"] = labels
        features.pop("offset_mapping")
        return features

    def encode_tags(self, labels: List[List[int]], encodings: BatchEncoding) -> List[List[int]]:
        """Encode tags to fix mismatch caused by token split into multiple subtokens by tokenizer.
        Special tokens have a word id that is None.
        We set the label to -100 so they are automatically ignored in the loss function.
        We set the label for the first token of each word.
        For the other tokens in a word, we set the label to either the current label or -100,
        depending on the label_all_tokens flag.
        Source: https://github.com/PyTorchLightning/lightning-transformers/blob/fc4703498a057476205dd4e518f8fcd09654c31b/lightning_transformers/task/nlp/token_classification/data.py"""
        encoded_labels = []
        for i, label in enumerate(labels):
            word_ids = encodings.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:  # special token = -100 (IGNORE_INDEX)
                    label_ids.append(self.IGNORE_INDEX)
                elif word_idx != previous_word_idx:  # first token of the word = label
                    label_ids.append(label[word_idx])
                else:  # other tokens = label or -100 depending on the label_all_tokens flag
                    label_ids.append(
                        label[word_idx] if self.label_all_tokens else self.IGNORE_INDEX
                    )
                previous_word_idx = word_idx
            encoded_labels.append(label_ids)
        return encoded_labels

    def _class_encode_column(self, column_name: str) -> None:
        new_features = self.dataset["train"].features.copy()
        new_features[column_name] = HFSequence(
            feature=ClassLabel(num_classes=self.num_classes, names=self.target_names)
        )
        self.dataset = self.dataset.cast(new_features)

    @property
    def collate_fn(self) -> Optional[Callable[[Any], Any]]:
        if self.processing_batch_size and self.processing_batch_size > 0:
            # ignoring type to avoid unexpected tokenizer argument defined in parent dataclass
            data_collator = CustomDataCollatorForTokenClassification(tokenizer=self.tokenizer)  # type: ignore
            assert callable(data_collator)
            return data_collator
        return None

    def id2str(self, int_: int) -> str:
        if self.dataset is None:
            raise AttributeError("Dataset has not been setup")
        str_ = self.dataset["train"].features["labels"].feature.int2str(int_)
        assert isinstance(str_, str)
        return str_

    def str2id(self, str_: str) -> int:
        if self.dataset is None:
            raise AttributeError("Dataset has not been setup")
        int_ = self.dataset["train"].features["labels"].feature.str2int(str_)
        assert isinstance(int_, int)
        return int_
