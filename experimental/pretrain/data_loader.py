import linecache
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data.dataset import Dataset
from transformers import DataCollatorForLanguageModeling


@dataclass
class DataCollatorForLazyLanguageModeling(DataCollatorForLanguageModeling):
    block_size: int = 512

    def __call__(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch, attention_mask = self._tensorize_batch(examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            label_key = "masked_lm_labels"
        else:
            inputs, labels = batch, batch
            label_key = "labels"
        labels = labels.masked_fill(attention_mask == 0, -100)
        return {"input_ids": batch, label_key: labels}

    def _tensorize_batch(self, examples: List[str]) -> torch.Tensor:

        if self.tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({self.tokenizer.__class__.__name__}) does not have one."
            )

        tensor_examples = self.tokenizer.batch_encode_plus(
            [ex for ex in examples if ex],
            max_length=self.block_size,
            return_tensors="pt",
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True,
        )

        input_ids, attention_mask = tensor_examples["input_ids"], tensor_examples["attention_mask"]
        return input_ids, attention_mask


class LazyLineByLineTextDataset(Dataset):
    """
    Credit: @bramvanroy for this linecache implementation.
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.num_entries = self._get_n_lines(self.file_path)

    @staticmethod
    def _get_n_lines(fin, size=65536):
        # borrowed from https://stackoverflow.com/a/9631635/1150683
        def blocks(files):
            while True:
                b = files.read(size)
                if not b:
                    break
                yield b

        with open(fin, encoding="utf-8") as fhin:
            n_lines = sum(bl.count("\n") for bl in blocks(fhin))
        return n_lines

    def __getitem__(self, idx):
        """
        :param idx (int): the index of the line to get
        :return (str or None): The line as a string (newline removed) or None if there is an exception.
        """
        # linecache starts counting from one, not zero, +1 the given index
        return linecache.getline(self.file_path, idx + 1).rstrip()

    def __len__(self):
        return self.num_entries
