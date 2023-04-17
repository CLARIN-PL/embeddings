import typing
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import datasets
import pandas as pd
from datasets import DatasetDict
from datasets.utils import Version
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BatchEncoding

from embeddings.data.datamodule import HuggingFaceDataModule, HuggingFaceDataset
from embeddings.data.io import T_path


class CharToTokenMapper:
    @staticmethod
    def _get_tokens_ids(sequence_ids: List[Any], input_len: int) -> Tuple[int, int]:
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = input_len - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        return token_start_index, token_end_index

    @staticmethod
    def _get_answer_start_and_end_char(answers: Dict[str, Any]) -> Tuple[int, int]:
        start_char = (
            answers["answer_start"][0]
            if isinstance(answers["answer_start"], list)
            else answers["answer_start"]
        )
        end_char = start_char + len(
            answers["text"][0] if isinstance(answers["text"], list) else answers["text"]
        )
        return start_char, end_char

    @staticmethod
    def _add_cls_tokens(features: BatchEncoding, cls_index: str) -> BatchEncoding:
        features["start_positions"].append(cls_index)
        features["end_positions"].append(cls_index)
        return features

    @staticmethod
    def get_token_positions_train(
        features: BatchEncoding, all_answers: Any, tokenizer: AutoTokenizer
    ) -> BatchEncoding:
        """Based on: https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py"""

        sample_mapping = features["overflow_to_sample_mapping"].copy()
        offset_mapping = features["offset_mapping"].copy()

        features["start_positions"] = []
        features["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = features["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = features.sequence_ids(i)

            sample_index = sample_mapping[i]
            answers = all_answers[sample_index]

            if not answers or (not answers["text"] and not answers["answer_start"]):
                features = CharToTokenMapper._add_cls_tokens(features, cls_index=cls_index)

            else:
                start_char, end_char = CharToTokenMapper._get_answer_start_and_end_char(
                    answers=answers
                )
                token_start_index, token_end_index = CharToTokenMapper._get_tokens_ids(
                    sequence_ids=sequence_ids, input_len=len(input_ids)
                )

                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    features = CharToTokenMapper._add_cls_tokens(features, cls_index=cls_index)
                else:
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    features["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    features["end_positions"].append(token_end_index + 1)
        return features


class QuestionAnsweringDataModule(HuggingFaceDataModule):
    def __init__(
        self,
        dataset_name_or_path: T_path,
        tokenizer_name_or_path: T_path,
        max_seq_length: Optional[int],
        train_batch_size: int,
        eval_batch_size: int,
        doc_stride: int,
        target_field: str = "answers",
        question_field: str = "question",
        context_field: str = "context",
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        batch_encoding_kwargs: Optional[Dict[str, Any]] = None,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        seed: int = 441,
        **kwargs: Any
    ) -> None:
        self.question_field = question_field
        self.context_field = context_field
        self.doc_stride = doc_stride
        self.batch_encoding_kwargs = batch_encoding_kwargs
        self.overflow_to_sample_mapping: Dict[str, List[List[List[int]]]] = {}
        self.offset_mapping: Dict[str, List[int]] = {}
        self.dataset_raw: Optional[datasets.DatasetDict] = None
        self.processed_data_stage: Optional[str] = None

        self.dataset = datasets.DatasetDict(
            {
                "train": datasets.Dataset.from_pandas(
                    pd.DataFrame(), info=datasets.DatasetInfo(version="0.0.0")
                )
            }
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
            dataloader_kwargs=dataloader_kwargs,
            seed=seed,
            **kwargs,
        )
        if isinstance(self.dataset_raw, datasets.DatasetDict):
            self.splits: List[str] = list(self.dataset_raw.keys())
        else:
            self.splits = ["train", "validation"]
        self.process_data(stage="fit")

    @typing.overload
    def process_data(self) -> None:
        pass

    @typing.overload
    def process_data(self, stage: Optional[str] = None) -> None:
        pass

    def process_data(self, stage: Optional[str] = None) -> None:
        assert isinstance(self.dataset_raw, datasets.DatasetDict)
        self.dataset = deepcopy(self.dataset_raw)
        if stage == "fit":
            self.dataset = DatasetDict(
                {k: v for k, v in self.dataset.items() if k in {"train", "validation"}}
            )

        if stage is None:
            return

        columns = [c for c in self.dataset["train"].column_names if c not in self.LOADER_COLUMNS]

        for split in self.dataset.keys():
            if split in {"train", "validation"} and stage == "fit":
                self.dataset[split] = self.dataset[split].map(
                    self.convert_to_features_train,
                    batched=True,
                    batch_size=self.processing_batch_size,
                    remove_columns=columns,
                )
            else:
                self.dataset[split] = self.dataset[split].map(
                    self.convert_to_features,
                    batched=True,
                    batch_size=self.processing_batch_size,
                    remove_columns=columns,
                )

            self.overflow_to_sample_mapping[split] = self.dataset[split][
                "overflow_to_sample_mapping"
            ]
            self.offset_mapping[split] = self.dataset[split]["offset_mapping"]
            self.dataset[split] = self.dataset[split].remove_columns(
                ["offset_mapping", "overflow_to_sample_mapping"]
            )

        self.dataset.set_format(type="torch")

    def prepare_data(self) -> None:
        AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)

    def convert_to_features_train(
        self, example_batch: Dict[str, Any], indices: Optional[List[int]] = None
    ) -> BatchEncoding:
        features = self.tokenizer(
            example_batch[self.question_field],
            example_batch[self.context_field],
            stride=self.doc_stride,
            max_length=self.max_seq_length,
            **self.batch_encoding_kwargs,
        )
        features = CharToTokenMapper.get_token_positions_train(
            features, example_batch[self.target_field], tokenizer=self.tokenizer
        )
        return features

    def convert_to_features(
        self, example_batch: Dict[str, Any], indices: Optional[List[int]] = None
    ) -> BatchEncoding:
        features = self.tokenizer(
            example_batch[self.question_field],
            example_batch[self.context_field],
            stride=self.doc_stride,
            max_length=self.max_seq_length,
            **self.batch_encoding_kwargs,
        )
        return features

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.has_setup:
            self.dataset_raw = self.load_dataset()
            if not self.dataset_raw["train"].info.version:
                self.dataset_raw["train"].info.version = Version("0.0.1")
            self.data_loader_has_setup = True
        if self.processed_data_stage and (self.processed_data_stage != stage):
            self.process_data(stage=stage)
            self.processed_data_stage = stage

    @property
    def collate_fn(self) -> Optional[Callable[[Any], Any]]:
        return None

    def prepare_labels(self) -> None:
        return None

    def _class_encode_column(self, column_name: str) -> None:
        pass

    def test_dataloader(self) -> DataLoader[HuggingFaceDataset]:
        if "test" in self.splits and not "test" in self.dataset.keys():
            self.process_data(stage="test")
        return super().test_dataloader()
