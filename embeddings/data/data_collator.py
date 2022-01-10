from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from transformers import BatchEncoding, DataCollatorForTokenClassification


# ignoring `Class cannot subclass "DataCollatorForTokenClassification" (has type "Any")` error
@dataclass
class CustomDataCollatorForTokenClassification(DataCollatorForTokenClassification):  # type: ignore
    """
    Fix issue with original DataCollator which fails when creating BatchEncoding tensor with labels.

    Compared to the original version, the labels are removed from the dictionary (features object)
    that is passed to the tokenizer for padding because it does not pad the labels.
    When the tokenizer returns BatchEncoding object tries to convert the whole dictionary to tensor
    and fails.
    """

    label_name: str = "labels"

    def torch_call(self, features: List[Dict[str, Any]]) -> Union[BatchEncoding, Dict[str, Any]]:
        labels = [feature[self.label_name] for feature in features]
        features = [
            {k: v for k, v in feature_dict.items() if k != self.label_name}
            for feature_dict in features
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
        if self.tokenizer.padding_side == "right":
            batch[self.label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in labels
            ]
        else:
            batch[self.label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label)
                for label in labels
            ]

        return {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
