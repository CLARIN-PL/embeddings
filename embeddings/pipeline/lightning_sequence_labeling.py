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
    task_train_kwargs = {"devices": "auto", "accelerator": "auto"}
    task_model_kwargs = {"use_scheduler": True}
    datamodule_kwargs = {"max_seq_length": None}
    model_config_kwargs = {"classifier_dropout": None}

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
        model_config_kwargs: Optional[Dict[str, Any]] = None,
        predict_subset: LightingDataModuleSubset = LightingDataModuleSubset.TEST,
    ):
        self.task_train_kwargs.update(task_train_kwargs if task_train_kwargs else {})
        self.task_model_kwargs.update(task_model_kwargs if task_model_kwargs else {})
        self.datamodule_kwargs.update(datamodule_kwargs if datamodule_kwargs else {})
        self.model_config_kwargs.update(model_config_kwargs if model_config_kwargs else {})

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
            **self.datamodule_kwargs
        )
        trainer = pl.Trainer(default_root_dir=output_path, **self.task_train_kwargs)

        task = SequenceLabeling(
            model_name_or_path=embedding_name,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            finetune_last_n_layers=finetune_last_n_layers,
            config_kwargs=self.model_config_kwargs,
            task_model_kwargs=self.task_model_kwargs,
        )
        model = LightningModel(trainer=trainer, task=task, predict_subset=predict_subset)
        evaluator = SequenceLabelingEvaluator(
            evaluation_mode=evaluation_mode, tagging_scheme=tagging_scheme
        )
        super().__init__(datamodule, model, evaluator)
