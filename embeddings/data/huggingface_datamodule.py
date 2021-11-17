import abc
from typing import Any, Dict, List, Optional, TypeVar, Union

import datasets
import pytorch_lightning as pl
from datasets import ClassLabel, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BatchEncoding

from embeddings.utils.loggers import get_logger

HuggingFaceDataset = TypeVar("HuggingFaceDataset")

_logger = get_logger(__name__)


class HuggingFaceDataModule(pl.LightningDataModule):
    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        dataset_name: str,
        target_field: str,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
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
        self.dataset: DatasetDict
        self.num_labels: int

    @property
    def input_dim(self) -> int:
        # TODO: hardcoded for now because text pairs are encoded together
        return 768

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
        self.process_data(stage)

    def process_data(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            for split in self.dataset.keys():
                # TODO: remove after testing
                self.dataset[split] = self.dataset[split].filter(
                    lambda example, index: index % 5 == 0, with_indices=True
                )
                self.dataset[split] = self.dataset[split].map(
                    self.convert_to_features,
                    batched=True,
                    remove_columns=[self.target_field],
                )
                self.columns = [
                    c for c in self.dataset[split].column_names if c in self.loader_columns
                ]
                self.dataset[split].set_format(type="torch", columns=self.columns)

    def train_dataloader(self) -> DataLoader[HuggingFaceDataset]:
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size)

    def val_dataloader(self) -> DataLoader[HuggingFaceDataset]:
        if "validation" in self.dataset:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        # else:
        #     raise AttributeError("Validation dataset not available")

    def test_dataloader(self) -> DataLoader[HuggingFaceDataset]:
        if "test" in self.dataset:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        else:
            raise AttributeError("Test dataset not available")

    @abc.abstractmethod
    def initalize(self) -> None:
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
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        if isinstance(text_fields, str):
            text_fields = [text_fields]
        assert 1 <= len(text_fields) <= 2
        self.text_fields = text_fields
        super().__init__(
            model_name_or_path,
            dataset_name,
            target_field,
            max_seq_length,
            train_batch_size,
            eval_batch_size,
            load_dataset_kwargs,
            **kwargs,
        )

    def initalize(self) -> None:
        self.dataset = datasets.load_dataset(self.dataset_name, **self.load_dataset_kwargs)
        self.num_labels = len(set(ex[self.target_field] for ex in self.dataset["train"]))
        self.initialized = True

    def convert_to_features(
        self, example_batch: Dict[str, Any], indices: Optional[List[int]] = None
    ) -> BatchEncoding:
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(
                zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]])
            )
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            truncation=True,
        )

        features["labels"] = example_batch[self.target_field]

        return features


class SequenceLabelingDataModule(HuggingFaceDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        dataset_name: str,
        text_field: str,
        target_field: str,
        max_seq_length: int = 128,
        label_all_tokens: bool = False,
        train_batch_size: int = 4,
        eval_batch_size: int = 4,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self.text_field = text_field
        self.label_all_tokens = label_all_tokens
        self.label_to_id: Dict[str, int] = {}
        self.id_to_label: Dict[int, str] = {}
        super().__init__(
            model_name_or_path,
            dataset_name,
            target_field,
            max_seq_length,
            train_batch_size,
            eval_batch_size,
            load_dataset_kwargs,
            **kwargs,
        )

    def initalize(self) -> None:
        self.dataset = datasets.load_dataset(self.dataset_name, **self.load_dataset_kwargs)
        self.prepare_labels()
        self.initialized = True

    def prepare_labels(self) -> None:
        if isinstance(self.dataset["train"].features[self.target_field].feature, ClassLabel):
            labels = self.dataset["train"].features[self.target_field].feature.names
            # No need to convert the labels since they are already ints.
        else:
            # Create unique label set from train dataset.
            labels = sorted(
                {label for column in self.dataset["train"][self.target_field] for label in column}
            )
        self.num_labels = len(labels)
        for i, label in enumerate(labels):
            self.label_to_id[label] = i
            self.id_to_label[i] = label

    def convert_to_features(
        self, example_batch: Dict[str, Any], indices: Optional[List[int]] = None
    ) -> BatchEncoding:
        texts = example_batch[self.text_field]

        features = self.tokenizer.batch_encode_plus(
            texts,
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            is_split_into_words=True,
            return_offsets_mapping=True,
        )

        labels = self.encode_tags(labels=example_batch[self.target_field], encodings=features)
        features["labels"] = labels

        features.pop("offset_mapping")
        return features

    def encode_tags(self, labels: List[List[int]], encodings: BatchEncoding) -> List[List[int]]:
        """Encode tags to fix mismatch caused by token split into multiple subtokens by tokenizer.

        Source: https://github.com/PyTorchLightning/lightning-transformers/blob/fc4703498a057476205dd4e518f8fcd09654c31b/lightning_transformers/task/nlp/token_classification/data.py"""
        encoded_labels = []

        for i, label in enumerate(labels):
            word_ids = encodings.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if self.label_all_tokens else -100)
                previous_word_idx = word_idx

            encoded_labels.append(label_ids)

        return encoded_labels
