"""TODO: Refactor"""
import abc
import os
import sys
import warnings
from collections import ChainMap, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import datasets
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import Dataset, DatasetDict
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from embeddings.data.datamodule import HuggingFaceDataModule
from embeddings.data.io import T_path
from embeddings.model.lightning_module.huggingface_module import HuggingFaceLightningModule
from embeddings.task.lightning_task.lightning_task import LightningTask
from embeddings.utils.loggers import LightningLoggingConfig

SQUAD_V2_PREDICTED_ANSWER_TYPE = Dict[str, Union[str, int, float]]
SQUAD_V2_GOLD_ANSWER_TYPE = Dict[str, Union[Dict[str, Union[List[str], List[Any]]], str, int]]


HuggingFaceDataset = Type[Dataset]  # to refactor


class PretrainedQAModel(pl.LightningModule):  # type: ignore
    """
    TODO:
    Refactor:
    - Move module to appropiate path
    embeddings/model/lightning_module/question_answering.py

    Refactor pt. 2
    Refactor in separate PR (create seperate task for inference)
    https://github.com/CLARIN-PL/embeddings/issues/279
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.trainer = pl.Trainer(devices="auto", accelerator="auto")

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Dict[str, Union[QuestionAnsweringModelOutput, Dict[str, torch.Tensor]]]:
        outputs = self.model(**batch)
        return {"data": batch, "outputs": outputs}

    def predict(
        self,
        dataloaders: Union[
            DataLoader[HuggingFaceDataset], Sequence[DataLoader[HuggingFaceDataset]]
        ],
    ) -> List[Dict[str, Union[QuestionAnsweringModelOutput, Dict[str, torch.Tensor]]]]:
        assert self.trainer is not None
        return self.trainer.predict(  # type: ignore
            model=self,
            dataloaders=dataloaders,
            return_predictions=True,
        )


class QuestionAnsweringModule(HuggingFaceLightningModule[AutoModelForQuestionAnswering]):
    """
    TODO:
    Refactor
    - Move module to appropiate path
      embeddings/model/lightning_module/question_answering.py

    Refactor pt. 2:
    - Refactor functions `setup`, `forward` `freeze_transformer`
    """

    downstream_model_type = AutoModelForQuestionAnswering

    def __init__(
        self,
        model_name_or_path: T_path,
        finetune_last_n_layers: int,
        config_kwargs: Optional[Dict[str, Any]] = None,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__(metrics=None, **task_model_kwargs if task_model_kwargs else {})
        self.save_hyperparameters({"downstream_model_type": self.downstream_model_type.__name__})
        self.downstream_model_type = self.downstream_model_type
        self.config_kwargs = config_kwargs if config_kwargs else {}
        self.target_names: Optional[List[str]] = None
        self._init_model()

    def _init_metrics(self) -> None:
        pass

    def _init_model(self) -> None:
        self.config = AutoConfig.from_pretrained(
            self.hparams.model_name_or_path,
            **self.config_kwargs,
        )
        self.model: AutoModel = self.downstream_model_type.from_pretrained(
            self.hparams.model_name_or_path, config=self.config
        )
        if self.hparams.finetune_last_n_layers > -1:
            self.freeze_transformer(finetune_last_n_layers=self.hparams.finetune_last_n_layers)

    def freeze_transformer(self, finetune_last_n_layers: int) -> None:
        """Borrowed from clarinpl-embeddings library"""
        if finetune_last_n_layers == 0:
            for name, param in self.model.base_model.named_parameters():
                param.requires_grad = False
        else:
            no_layers = self.model.config.num_hidden_layers
            for name, param in self.model.base_model.named_parameters():
                if name.startswith("embeddings"):
                    layer = 0
                elif name.startswith("encoder"):
                    layer = int(name.split(".")[2])
                elif name.startswith("pooler"):
                    layer = sys.maxsize
                else:
                    raise ValueError("Parameter name not recognized when freezing transformer")
                if layer >= (no_layers - finetune_last_n_layers):
                    break
                param.requires_grad = False

    def setup(self, stage: Optional[str] = None) -> None:
        """Borrowed from clarinpl-embeddings library"""
        if stage in ("fit", None):
            assert self.trainer is not None
            if self.hparams.use_scheduler:
                train_loader = self.trainer.datamodule.train_dataloader()
                gpus = getattr(self.trainer, "gpus") if getattr(self.trainer, "gpus") else 0
                tb_size = self.hparams.train_batch_size * max(1, gpus)
                ab_size = tb_size * self.trainer.accumulate_grad_batches
                self.total_steps: int = int(
                    (len(train_loader.dataset) / ab_size) * float(self.trainer.max_epochs)
                )
            self._init_metrics()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Borrowed from clarinpl-embeddings library"""

        assert (not (args and kwargs)) and (args or kwargs)
        inputs = kwargs if kwargs else args
        if isinstance(inputs, tuple):
            inputs = dict(ChainMap(*inputs))
        return self.model(**inputs)

    def shared_step(self, **batch: Any) -> Any:
        outputs = self(**batch)
        return {"data": batch, "outputs": outputs}

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        batch, batch_idx = args
        outputs = self(**batch)

        self.log("train/Loss", outputs.loss)
        if self.hparams.use_scheduler:
            assert self.trainer is not None
            last_lr = self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()
            self.log("train/BaseLR", last_lr[0], prog_bar=True)
            self.log("train/LambdaLR", last_lr[1], prog_bar=True)
        return {"loss": outputs.loss}

    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        batch, batch_idx = args
        outputs = self(**batch)
        self.log("val/Loss", outputs.loss)
        return {"loss": outputs.loss}

    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        batch, batch_idx = args
        _ = self.shared_step(**batch)
        return None

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        batch, batch_idx = args
        return self.shared_step(**batch)

    def training_epoch_end(self, outputs: List[Any]) -> None:
        pass

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        pass

    def test_epoch_end(self, outputs: List[Any]) -> None:
        pass


class QuestionAnsweringTask(LightningTask):
    """
    TODO:
    Refactor
     embeddings/task/lightning_task/question_answering.py

     Refactor pt 2:
     - Drop MlFlow logger as required parameter
     - Add postprocess directly to task
    """

    def __init__(
        self,
        model_name_or_path: T_path,
        output_path: T_path,
        model_config_kwargs: Dict[str, Any],
        task_model_kwargs: Dict[str, Any],
        task_train_kwargs: Dict[str, Any],
        early_stopping_kwargs: Dict[str, Any],
        model_checkpoint_kwargs: Dict[str, Any],
        mlflow_logger: MLFlowLogger,
        finetune_last_n_layers: int = -1,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
        # disable a thousand of warnings of HF
        warnings.simplefilter("ignore")
        # set os environ variable for multiprocesses
        os.environ["PYTHONWARNINGS"] = "ignore"

        super().__init__(
            output_path=output_path,
            task_train_kwargs=task_train_kwargs,
            early_stopping_kwargs=early_stopping_kwargs,
            model_checkpoint_kwargs=model_checkpoint_kwargs,
            logging_config=LightningLoggingConfig.from_flags(),
        )
        self.model_name_or_path = model_name_or_path
        self.model_config_kwargs = model_config_kwargs
        self.task_model_kwargs = task_model_kwargs
        self.finetune_last_n_layers = finetune_last_n_layers
        self.task_train_kwargs = task_train_kwargs
        self.mlflow_logger = mlflow_logger

    def build_task_model(self) -> None:
        self.model = QuestionAnsweringModule(
            model_name_or_path=self.model_name_or_path,
            finetune_last_n_layers=self.finetune_last_n_layers,
            config_kwargs=self.model_config_kwargs,
            task_model_kwargs=self.task_model_kwargs,
        )

    def fit(
        self,
        data: HuggingFaceDataModule,
        run_name: Optional[str] = None,
    ) -> None:
        if not self.model:
            raise self.MODEL_UNDEFINED_EXCEPTION

        callbacks = None
        if self.early_stopping_kwargs:
            callbacks = [EarlyStopping(**self.early_stopping_kwargs)]

        self.trainer = pl.Trainer(
            default_root_dir=str(self.output_path),
            callbacks=callbacks,
            logger=self.mlflow_logger,
            **self.task_train_kwargs,
        )
        try:
            self.trainer.fit(self.model, data)
        except Exception as e:
            del self.trainer
            torch.cuda.empty_cache()
            raise e

    def predict(self, dataloader: Any, return_names: bool = True) -> Any:
        assert self.model is not None
        assert self.trainer is not None
        return self.trainer.predict(model=self.model, dataloaders=dataloader)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: T_path,
        output_path: T_path,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
        logging_config: Optional[LightningLoggingConfig] = None,
    ) -> "LightningTask":
        return cls.restore_task_model(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            task_train_kwargs=task_train_kwargs,
            lightning_module=QuestionAnsweringModule,
            logging_config=logging_config,
        )


def unwrap_outputs_from_batches(
    predictions: List[Dict[str, Union[QuestionAnsweringModelOutput, Dict[str, torch.Tensor]]]]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    TODO: Refactor
    Move to embedding/transformation/lightning_transformation/question_answering_output_transformation.py
    """
    tensors_lists_dict: Dict[str, Dict[str, List[torch.Tensor]]] = {}

    for batch_id, batch in enumerate(predictions):
        for key in batch.keys():
            if key not in tensors_lists_dict.keys():
                tensors_lists_dict[key] = defaultdict(list)
            for tensor_key, tensor in batch[key].items():
                if tensor_key == "loss":
                    continue
                assert isinstance(tensor, torch.Tensor)
                if tensor.dtype in {torch.bfloat16, torch.float16}:
                    tensor = tensor.to(dtype=torch.float32)
                tensors_lists_dict[key][tensor_key].append(tensor)

    output: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)

    for key in tensors_lists_dict.keys():
        for tensor_key, tensors in tensors_lists_dict[key].items():
            output[key][tensor_key] = torch.cat(tensors)

    return output


class QABasePostprocessor(abc.ABC):
    """
    TODO: Refactor
    Move to embedding/transformation/lightning_transformation/question_answering_output_transformation.py

    Refactor pt 2:
    - Add transformation as base class, and change name of postprocess to transformer
    """
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
    TODO:
    Move to embedding/transformation/lightning_transformation/question_answering_output_transformation.py

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
        softmax_scores = torch.functional.F.softmax(scores)  # type: ignore
        for prob, pred in zip(softmax_scores, predictions):
            pred["softmax_score"] = prob

        return sorted(predictions, key=lambda x: x["softmax_score"], reverse=True)  # type: ignore

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

            predictions.extend(
                self._get_topk_not_cls_predictions_from_output(
                    start_logits=start_logits[1:],
                    end_logits=end_logits[1:],
                    offset_mapping=offset_mappings[output_index],
                )
            )

        predictions.append(min_no_answer_score)  # type: ignore
        predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)[  # type: ignore
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


def split_train_dev_test(
    ds: pd.DataFrame,
    train_size: float,
    dev_size: float,
    test_size: float,
    seed: int,
    context_column: str = "context",
    stratify_column: Optional[str] = None,
) -> DatasetDict:
    """
    TODO: Move to embeddings/transformation/hf_transformation/qa_data_split_transformation.py
    Refactor pt 2
    Q&A require separate split_train pipeline refactor it as a transformation and improve parametrization"
    """
    dataset = DatasetDict()
    unique_contexts = list(sorted(ds[context_column].unique()))
    context_ids = list(range(len(unique_contexts)))
    contexts_mapping = dict(zip(unique_contexts, context_ids))
    ds["context_id"] = ds.apply(lambda x: contexts_mapping[x[context_column]], axis=1)

    stratify = None
    if stratify_column:
        assert stratify_column in ds.columns
        stratify = []
        for context_id in context_ids:
            df = ds[ds.context_id == context_id]
            value = df[stratify_column].values
            assert len(value.unique()) == 1
            stratify.append(value.unique()[0])

    train_indices, validation_indices = train_test_split(
        context_ids,
        train_size=train_size,
        stratify=stratify,
        random_state=seed,
    )

    dataset["train"] = Dataset.from_pandas(
        ds[ds.context_id.isin(train_indices)], preserve_index=False
    )

    if dev_size and test_size:
        dev_indices, test_indices = train_test_split(
            validation_indices,
            train_size=round(dev_size / (1 - train_size), 2),
            random_state=seed,
        )

        dataset["validation"] = Dataset.from_pandas(
            ds[ds.context_id.isin(dev_indices)], preserve_index=False
        )
        dataset["test"] = Dataset.from_pandas(
            ds[ds.context_id.isin(test_indices)], preserve_index=False
        )

    elif dev_size and not test_size:
        dataset["validation"] = Dataset.from_pandas(
            ds[ds.context_id.isin(validation_indices)], preserve_index=False
        )

    elif test_size and not dev_size:
        dataset["test"] = Dataset.from_pandas(
            ds[ds.context_id.isin(validation_indices)], preserve_index=False
        )

    return dataset


class QAMetric(abc.ABC):
    """
    TODO:
    Refactor:
    embeddings/metric/question_answering.py

    Refactor pt 2:
    - Decide whether we need additional seperate base QA metric class
    """

    """TODO: Refactor it as metric"""
    @abc.abstractmethod
    def calculate(
        self,
        predictions: List[SQUAD_V2_PREDICTED_ANSWER_TYPE],
        references: List[SQUAD_V2_GOLD_ANSWER_TYPE],
    ) -> Dict[str, Union[float, int]]:
        pass


class SQUADv2Metric(QAMetric):
    """
    TODO:
    Refactor:
    embeddings/metric/question_answering.py
    """

    def __init__(self, no_answer_threshold: float = 1.0) -> None:
        self.metric = datasets.load_metric("squad_v2")
        self.no_answer_threshold = no_answer_threshold

    def calculate(
        self,
        predictions: List[SQUAD_V2_PREDICTED_ANSWER_TYPE],
        references: List[SQUAD_V2_GOLD_ANSWER_TYPE],
    ) -> Dict[Any, Any]:
        metrics = self.metric.compute(
            predictions=predictions,
            references=references,
            no_answer_threshold=self.no_answer_threshold,
        )
        assert metrics is not None
        return metrics


class QAEvaluator(abc.ABC):
    """
    TODO:
    Refactor:
    embeddings/evaluator/qa_evaluator.py
    Refactor pt 2:
    Rewrite it as evaluator
    """
    @abc.abstractmethod
    def evaluate(self, scores: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass


class QASquadV2Evaluator(QAEvaluator):
    """TODO:
    Refactor:
    embeddings/evaluator/qa_evaluator.py
    Refactor pt 2:
    Rewrite it as evaluator"""
    def __init__(self, no_answer_threshold: float = 1.0):
        self.metric = SQUADv2Metric(no_answer_threshold=no_answer_threshold)
        self.postprocessor = QAPredictionPostProcessor()

    def evaluate(self, scores: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        metrics = {}
        outputs = {}

        for split in tqdm(scores.keys(), desc="Split"):
            outputs[split] = self.postprocessor.postprocess(**scores[split])

            references: List[SQUAD_V2_GOLD_ANSWER_TYPE] = []
            for example_id, example in enumerate(outputs[split]):
                references.append(
                    {
                        "id": example_id,
                        "answers": {
                            "answer_start": example["answers"]["answer_start"]
                            if example["answers"]
                            else [],
                            "text": example["answers"]["text"] if example["answers"] else [],
                        },
                    }
                )
            predictions: List[SQUAD_V2_PREDICTED_ANSWER_TYPE] = [
                {"id": it_id, **it["predicted_answer"]} for it_id, it in enumerate(outputs[split])
            ]
            metrics[split] = SQUADv2Metric().calculate(
                predictions=predictions, references=references
            )

        return metrics, outputs
