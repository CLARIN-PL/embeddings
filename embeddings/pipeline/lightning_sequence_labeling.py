from typing import Any, Dict, Optional

import datasets
import pytorch_lightning as pl
from numpy import typing as nptyping

from embeddings.data.datamodule import SequenceLabelingDataModule
from embeddings.data.dataset import LightingDataModuleSubset
from embeddings.data.io import T_path
from embeddings.evaluator.sequence_labeling_evaluator import (
    EvaluationMode,
    SequenceLabelingEvaluator,
    TaggingScheme,
)
from embeddings.model.lightning_model import LightningModel
from embeddings.pipeline.lightning_pipeline import LightningPipeline
from embeddings.task.lightning_task.sequence_labeling import SequenceLabeling


class LightningSequenceLabelingPipeline(
    LightningPipeline[datasets.DatasetDict, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]]
):
    DEFAULT_TASK_TRAIN_KWARGS = {"devices": "auto", "accelerator": "auto"}
    DEFAULT_DATAMODULE_KWARGS = {"max_seq_length": None}
    DEFAULT_TASK_MODEL_KWARGS = {"use_scheduler": False}

    def __init__(
        self,
        embedding_name: str,
        dataset_name_or_path: T_path,
        input_column_name: str,
        target_column_name: str,
        output_path: T_path,
        evaluation_mode: EvaluationMode = EvaluationMode.CONLL,
        tagging_scheme: Optional[TaggingScheme] = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        label_all_tokens: bool = False,
        finetune_last_n_layers: int = -1,
        tokenizer_name: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        datamodule_kwargs: Optional[Dict[str, Any]] = None,
        batch_encoding_kwargs: Optional[Dict[str, Any]] = None,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
    ):
        datamodule = SequenceLabelingDataModule(
            tokenizer_name_or_path=tokenizer_name if tokenizer_name else embedding_name,
            dataset_name_or_path=dataset_name_or_path,
            text_field=input_column_name,
            target_field=target_column_name,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            label_all_tokens=label_all_tokens,
            tokenizer_kwargs=tokenizer_kwargs,
            batch_encoding_kwargs=batch_encoding_kwargs,
            load_dataset_kwargs=load_dataset_kwargs,
            **datamodule_kwargs if datamodule_kwargs else self.DEFAULT_DATAMODULE_KWARGS,
        )
        trainer = pl.Trainer(
            default_root_dir=output_path,
            **task_train_kwargs if task_train_kwargs else self.DEFAULT_TASK_TRAIN_KWARGS,
        )

        task = SequenceLabeling(
            model_name_or_path=embedding_name,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            finetune_last_n_layers=finetune_last_n_layers,
            task_model_kwargs=task_model_kwargs
            if task_model_kwargs
            else self.DEFAULT_TASK_MODEL_KWARGS,
        )
        model = LightningModel(
            trainer=trainer, task=task, predict_subset=LightingDataModuleSubset.TEST
        )
        evaluator = SequenceLabelingEvaluator(
            evaluation_mode=evaluation_mode, tagging_scheme=tagging_scheme
        )
        super().__init__(datamodule, model, evaluator)
