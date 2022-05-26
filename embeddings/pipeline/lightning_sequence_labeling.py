from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets
from pytorch_lightning.accelerators import Accelerator

from embeddings.config.lightning_config import LightningBasicConfig, LightningConfig
from embeddings.data.datamodule import SequenceLabelingDataModule
from embeddings.data.dataset import LightingDataModuleSubset
from embeddings.data.io import T_path
from embeddings.evaluator.evaluation_results import Predictions, SequenceLabelingEvaluationResults
from embeddings.evaluator.sequence_labeling_evaluator import SequenceLabelingEvaluator
from embeddings.metric.sequence_labeling import EvaluationMode, TaggingScheme
from embeddings.model.lightning_model import LightningModel
from embeddings.pipeline.lightning_pipeline import LightningPipeline
from embeddings.task.lightning_task.sequence_labeling import SequenceLabelingTask
from embeddings.utils.json_dict_persister import JsonPersister
from embeddings.utils.loggers import LightningLoggingConfig


class LightningSequenceLabelingPipeline(
    LightningPipeline[datasets.DatasetDict, Predictions, SequenceLabelingEvaluationResults]
):
    def __init__(
        self,
        embedding_name_or_path: T_path,
        dataset_name_or_path: T_path,
        input_column_name: str,
        target_column_name: str,
        output_path: T_path,
        evaluation_filename: str = "evaluation.json",
        evaluation_mode: EvaluationMode = EvaluationMode.CONLL,
        tagging_scheme: Optional[TaggingScheme] = None,
        config: LightningConfig = LightningBasicConfig(),
        logging_config: LightningLoggingConfig = LightningLoggingConfig(),
        tokenizer_name_or_path: Optional[T_path] = None,
        devices: Optional[Union[List[int], str, int]] = "auto",
        accelerator: Optional[Union[str, Accelerator]] = "auto",
        predict_subset: LightingDataModuleSubset = LightingDataModuleSubset.TEST,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        model_checkpoint_kwargs: Optional[Dict[str, Any]] = None,
    ):
        task_train_kwargs = config.task_train_kwargs
        task_train_kwargs.update({"devices": devices, "accelerator": accelerator})
        tokenizer_name_or_path = tokenizer_name_or_path or embedding_name_or_path
        model_checkpoint_kwargs = model_checkpoint_kwargs if model_checkpoint_kwargs else {}
        output_path = Path(output_path)

        datamodule = SequenceLabelingDataModule(
            tokenizer_name_or_path=tokenizer_name_or_path,
            dataset_name_or_path=dataset_name_or_path,
            text_field=input_column_name,
            target_field=target_column_name,
            train_batch_size=config.task_model_kwargs["train_batch_size"],
            eval_batch_size=config.task_model_kwargs["eval_batch_size"],
            tokenizer_kwargs=config.tokenizer_kwargs,
            batch_encoding_kwargs=config.batch_encoding_kwargs,
            load_dataset_kwargs=load_dataset_kwargs,
            dataloader_kwargs=config.dataloader_kwargs,
            **config.datamodule_kwargs
        )
        task = SequenceLabelingTask(
            model_name_or_path=embedding_name_or_path,
            output_path=output_path,
            num_classes=datamodule.num_classes,
            finetune_last_n_layers=config.finetune_last_n_layers,
            model_config_kwargs=config.model_config_kwargs,
            task_model_kwargs=config.task_model_kwargs,
            task_train_kwargs=task_train_kwargs,
            early_stopping_kwargs=config.early_stopping_kwargs,
            logging_config=logging_config,
            evaluation_mode=evaluation_mode,
            tagging_scheme=tagging_scheme,
            model_checkpoint_kwargs=model_checkpoint_kwargs,
        )
        model = LightningModel(task=task, predict_subset=predict_subset)
        evaluator = SequenceLabelingEvaluator(
            evaluation_mode=evaluation_mode, tagging_scheme=tagging_scheme
        ).persisting(JsonPersister(path=output_path.joinpath(evaluation_filename)))
        super().__init__(datamodule, model, evaluator, output_path, logging_config)
