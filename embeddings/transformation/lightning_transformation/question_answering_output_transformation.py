import abc
from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers.modeling_outputs import QuestionAnsweringModelOutput


def unwrap_outputs_from_batches(
    predictions: List[Dict[str, Union[QuestionAnsweringModelOutput, Dict[str, torch.Tensor]]]]
) -> Dict[str, Dict[str, torch.Tensor]]:
    tensors_lists_dict: Dict[str, Dict[str, List[torch.Tensor]]] = {}

    for batch_id, batch in enumerate(predictions):
        for key in batch.keys():
            if key not in tensors_lists_dict.keys():
                tensors_lists_dict[key] = defaultdict(list)
            for tensor_key, tensor in batch[key].items():
                if tensor_key == "loss":
                    continue
                if isinstance(tensor, np.ndarray):
                    tensor = torch.from_numpy(tensor)
                if tensor.dtype in {torch.bfloat16, torch.float16}:
                    tensor = tensor.to(dtype=torch.float32)
                tensors_lists_dict[key][tensor_key].append(tensor)

    output: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)

    for key in tensors_lists_dict.keys():
        for tensor_key, tensors in tensors_lists_dict[key].items():
            output[key][tensor_key] = torch.cat(tensors)

    return output


class QABasePostprocessor(abc.ABC):
    @abc.abstractmethod
    def postprocess(
        self,
        examples: Dataset,
        overflow_to_sample_mapping: List[int],
        offset_mappings: List[List[List[int]]],
        outputs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        pass


class QAPredictionPostProcessor(QABasePostprocessor):
    """
    Based on QA huggingface transformers pipeline.
    https://github.com/huggingface/transformers/blob/d6b8e9cec7301ba02f642588a6f12e78ec3b9798/examples/pytorch/question-answering/utils_qa.py#L31
    https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb#scrollTo=n9qywopnIrJH
    """

    def __init__(
        self,
        top_k_answers: int = 20,
        max_answer_length: int = 40,
        question_field_name: str = "question",
        context_field_name: str = "context",
        target_field_name: str = "answers",
    ) -> None:
        super().__init__()
        self.top_k_answers = top_k_answers
        self.max_answer_length = max_answer_length
        self.question_field_name = question_field_name
        self.context_field_name = context_field_name
        self.target_field_name = target_field_name

    def _get_topk_not_cls_predictions_from_output(
        self, start_logits: torch.Tensor, end_logits: torch.Tensor, offset_mapping: List[List[int]]
    ) -> List[Dict[str, Any]]:
        topk_start_indices = torch.topk(start_logits, self.top_k_answers).indices.tolist()
        topk_end_indices = torch.topk(end_logits, self.top_k_answers).indices.tolist()
        topk_predictions = []

        for start_index in topk_start_indices:
            start_index_offset = offset_mapping[start_index + 1]

            for end_index in topk_end_indices:
                end_index_offset = offset_mapping[end_index + 1]

                # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                # to part of the input_ids that are not in the context.
                if (
                    (start_index + 1) >= len(offset_mapping)
                    or (end_index + 1) >= len(offset_mapping)
                    or start_index_offset is None
                    or len(start_index_offset) < 2
                    or end_index_offset is None
                    or len(end_index_offset) < 2
                ):
                    continue

                # Don't consider answers with a length that is either < 0 or > max_answer_length.
                if end_index < start_index or end_index - start_index + 1 > self.max_answer_length:
                    continue

                topk_predictions.append(
                    {
                        "offsets": (start_index_offset[0], end_index_offset[1]),
                        "score": start_logits[start_index] + end_logits[end_index],
                        "start_logit": start_logits[start_index],
                        "end_logit": end_logits[end_index],
                        "start_token_index": start_index + 1,
                        "end_token_index": end_index + 1,
                    }
                )

        return topk_predictions

    @staticmethod
    def _get_predicted_text_from_context(
        context: str, predictions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        for prediction in predictions:
            offsets = prediction.pop("offsets")
            prediction["text"] = context[offsets[0] : offsets[1]]

        return predictions

    @staticmethod
    def _get_softmax_scores_with_sort(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scores = torch.from_numpy(np.array([pred.pop("score") for pred in predictions]))
        # Module torch.functional does not explicitly export attritube "F"
        softmax_scores = torch.functional.F.softmax(scores)  # type: ignore[attr-defined]
        for prob, pred in zip(softmax_scores, predictions):
            pred["softmax_score"] = prob
        # mypy thinks the function only returns Any
        return sorted(predictions, key=lambda x: x["softmax_score"], reverse=True)  # type: ignore[no-any-return]

    def _postprocess_example(
        self,
        example: Dict[str, Any],
        outputs: Dict[str, torch.Tensor],
        output_indices: List[int],
        offset_mappings: List[List[List[int]]],
    ) -> List[Dict[str, Any]]:
        min_no_answer_score = None
        predictions = []
        for output_index in output_indices:
            start_logits = outputs["start_logits"][output_index]
            end_logits = outputs["end_logits"][output_index]
            no_answer_score = start_logits[0] + end_logits[0]

            if min_no_answer_score is None or min_no_answer_score["score"] > no_answer_score:
                min_no_answer_score = {
                    "offsets": (0, 0),
                    "score": no_answer_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                    "start_token_index": 0,
                    "end_token_index": 0,
                }

            predictions += self._get_topk_not_cls_predictions_from_output(
                start_logits=start_logits[1:],
                end_logits=end_logits[1:],
                offset_mapping=offset_mappings[output_index],
            )
        # Argument 1 to "append" of "list" has incompatible type "Optional[Dict[str, object]]"; expected "Dict[str, Any]"
        predictions.append(min_no_answer_score)  # type: ignore[arg-type]
        # mypy thinks the function only returns Any
        predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)[  # type: ignore[no-any-return]
            : self.top_k_answers
        ]
        predictions = self._get_predicted_text_from_context(
            context=example[self.context_field_name], predictions=predictions
        )
        return self._get_softmax_scores_with_sort(predictions)

    def postprocess(
        self,
        examples: pd.DataFrame,
        overflow_to_sample_mapping: List[int],
        offset_mapping: List[List[List[int]]],
        outputs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        out = unwrap_outputs_from_batches(outputs)

        outputs_per_examples = defaultdict(list)
        for feature_index, example_id in enumerate(overflow_to_sample_mapping):
            outputs_per_examples[example_id].append(feature_index)

        processed = []
        for example_id, example in tqdm(
            examples.iterrows(), desc="Example", total=len(examples), leave=False
        ):
            example = example.to_dict()
            example_predictions = self._postprocess_example(
                example=example,
                outputs=out["outputs"],
                output_indices=outputs_per_examples[example_id],
                offset_mappings=offset_mapping,
            )
            best_answer = example_predictions[0]
            processed.append(
                {
                    "context": example[self.context_field_name],
                    "questions": example[self.question_field_name],
                    "answers": example[self.target_field_name],
                    "predicted_answer": {
                        "prediction_text": best_answer["text"],
                        "no_answer_probability": 0.0,
                    },
                    **{
                        k: v
                        for k, v in example.items()
                        if k not in ("context", "questions", "answers")
                    },
                }
            )

        return processed
