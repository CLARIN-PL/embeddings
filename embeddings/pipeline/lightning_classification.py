from typing import Any, Dict, Optional, Sequence, Union

import datasets
import pytorch_lightning as pl
from numpy import typing as nptyping

from embeddings.data.datamodule import TextClassificationDataModule
from embeddings.data.io import T_path
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.lightning_model import LightningModel
from embeddings.pipeline.lightning_pipeline import LightningPipeline
from embeddings.task.lightning_task.text_classification import TextClassification


class LightningClassificationPipeline(
    LightningPipeline[datasets.DatasetDict, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]]
):
    DEFAULT_TASK_TRAIN_KWARGS = {"devices": "auto", "accelerator": "auto"}
    DEFAULT_TASK_MODEL_KWARGS = {"use_scheduler": True}

    def __init__(
        self,
        embedding_name: str,
        dataset_name: str,
        input_column_name: Union[str, Sequence[str]],
        target_column_name: str,
        output_path: T_path,
        max_seq_length: Optional[int] = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        tokenizer_name: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        batch_encoding_kwargs: Optional[Dict[str, Any]] = None,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
    ):
        datamodule = TextClassificationDataModule(
            tokenizer_name_or_path=tokenizer_name if tokenizer_name else embedding_name,
            dataset_name=dataset_name,
            text_fields=input_column_name,
            target_field=target_column_name,
            max_seq_length=max_seq_length,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            tokenizer_kwargs=tokenizer_kwargs,
            batch_encoding_kwargs=batch_encoding_kwargs,
            load_dataset_kwargs=load_dataset_kwargs,
        )
        trainer = pl.Trainer(
            default_root_dir=output_path,
            **task_train_kwargs if task_train_kwargs else self.DEFAULT_TASK_TRAIN_KWARGS
        )

        task = TextClassification(
            model_name_or_path=embedding_name,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            task_model_kwargs=task_model_kwargs
            if task_model_kwargs
            else self.DEFAULT_TASK_MODEL_KWARGS,
        )
        model = LightningModel(trainer=trainer, task=task, predict_subset="test")
        evaluator = TextClassificationEvaluator()
        super().__init__(datamodule, model, evaluator)
