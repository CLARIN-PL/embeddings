from typing import Any, Dict, List, Optional, TypeVar, Union

import datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BatchEncoding

from embeddings.utils.loggers import get_logger

HuggingFaceDataset = TypeVar("HuggingFaceDataset")

_logger = get_logger(__name__)


class TextClassificationDataModule(pl.LightningDataModule):
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
        self.model_name_or_path = model_name_or_path
        self.dataset_name = dataset_name
        self.text_fields = text_fields
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

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.initialized:
            self.initalize()
        if stage == "fit":
            for split in self.dataset.keys():
                self.dataset[split] = self.dataset[split].map(
                    self.convert_to_features,
                    batched=True,
                    remove_columns=[self.target_field],
                )
                self.columns = [
                    c for c in self.dataset[split].column_names if c in self.loader_columns
                ]
                self.dataset[split].set_format(type="torch", columns=self.columns)

    def initalize(self) -> None:
        self.dataset = datasets.load_dataset(self.dataset_name, **self.load_dataset_kwargs)
        self.num_labels = len(set(ex[self.target_field] for ex in self.dataset["train"]))
        self.initialized = True

    def prepare_data(self) -> None:
        datasets.load_dataset(self.dataset_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

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

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            truncation=True,
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch[self.target_field]

        return features