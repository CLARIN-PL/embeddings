from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from transformers import BatchEncoding, DataCollatorForTokenClassification


@dataclass
class CustomDataCollatorForTokenClassification(DataCollatorForTokenClassification):
    label_name: str = "labels"

    def torch_call(self, features: List[Dict[str, Any]]) -> Union[BatchEncoding, Dict[str, Any]]:
        labels = [feature[self.label_name] for feature in features]
        features = [
            {k: v for k, v in feature_dict if k != self.label_name} for feature_dict in features
        ]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[self.label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in labels
            ]
        else:
            batch[self.label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label)
                for label in labels
            ]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch
