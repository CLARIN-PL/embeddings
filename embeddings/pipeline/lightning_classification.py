from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import datasets
from numpy import typing as nptyping

from embeddings.data.datamodule import TextClassificationDataModule
from embeddings.data.dataset import LightingDataModuleSubset
from embeddings.data.io import T_path
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.lightning_model import LightningModel
from embeddings.pipeline.lightning_pipeline import LightningPipeline
from embeddings.task.lightning_task.text_classification import TextClassificationTask
from embeddings.utils.json_dict_persister import JsonPersister
from embeddings.utils.utils import initialize_kwargs


class LightningClassificationPipeline(
    LightningPipeline[datasets.DatasetDict, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]]
):
    DEFAULT_TASK_TRAIN_KWARGS = {"devices": "auto", "accelerator": "auto"}
    DEFAULT_TASK_MODEL_KWARGS = {"use_scheduler": True}
    DEFAULT_DATAMODULE_KWARGS = {"max_seq_length": None}
    DEFAULT_MODEL_CONFIG_KWARGS = {"classifier_dropout": None}
    DEFAULT_EARLY_STOPPING_KWARGS = {"monitor": "val/Loss", "mode": "min", "patience": 3}

    def __init__(
        self,
        embedding_name_or_path: T_path,
        dataset_name_or_path: T_path,
        input_column_name: Union[str, Sequence[str]],
        target_column_name: str,
        output_path: T_path,
        evaluation_filename: str = "evaluation.json",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        finetune_last_n_layers: int = -1,
        tokenizer_name_or_path: Optional[T_path] = None,
        datamodule_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        batch_encoding_kwargs: Optional[Dict[str, Any]] = None,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
        model_config_kwargs: Optional[Dict[str, Any]] = None,
        early_stopping_kwargs: Optional[Dict[str, Any]] = None,
        predict_subset: LightingDataModuleSubset = LightingDataModuleSubset.TEST,
    ):
        self.datamodule_kwargs = initialize_kwargs(
            self.DEFAULT_DATAMODULE_KWARGS, datamodule_kwargs
        )
        self.task_train_kwargs = initialize_kwargs(
            self.DEFAULT_TASK_TRAIN_KWARGS, task_train_kwargs
        )
        self.model_config_kwargs = initialize_kwargs(
            self.DEFAULT_MODEL_CONFIG_KWARGS, model_config_kwargs
        )
        self.task_model_kwargs = initialize_kwargs(
            self.DEFAULT_TASK_MODEL_KWARGS, task_model_kwargs
        )
        self.early_stopping_kwargs = initialize_kwargs(
            self.DEFAULT_EARLY_STOPPING_KWARGS, early_stopping_kwargs
        )
        self.task_model_kwargs.update(
            {"train_batch_size": train_batch_size, "eval_batch_size": eval_batch_size}
        )
        tokenizer_name_or_path = (
            tokenizer_name_or_path if tokenizer_name_or_path else embedding_name_or_path
        )

        output_path = Path(output_path)
        datamodule = TextClassificationDataModule(
            tokenizer_name_or_path=tokenizer_name_or_path,
            dataset_name_or_path=dataset_name_or_path,
            text_fields=input_column_name,
            target_field=target_column_name,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            tokenizer_kwargs=tokenizer_kwargs,
            batch_encoding_kwargs=batch_encoding_kwargs,
            load_dataset_kwargs=load_dataset_kwargs,
            **self.datamodule_kwargs
        )
        task = TextClassificationTask(
            model_name_or_path=embedding_name_or_path,
            output_path=output_path,
            finetune_last_n_layers=finetune_last_n_layers,
            model_config_kwargs=self.model_config_kwargs,
            task_model_kwargs=self.task_model_kwargs,
            task_train_kwargs=self.task_train_kwargs,
            early_stopping_kwargs=self.early_stopping_kwargs,
        )
        model = LightningModel(task=task, predict_subset=predict_subset)
        evaluator = TextClassificationEvaluator().persisting(
            JsonPersister(path=output_path.joinpath(evaluation_filename))
        )
        super().__init__(datamodule, model, evaluator)
