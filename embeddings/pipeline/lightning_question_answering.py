from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets
import pandas as pd
from pytorch_lightning.accelerators import Accelerator
from tqdm import tqdm

from embeddings.config.lightning_config import LightningBasicConfig, LightningConfig
from embeddings.data.dataset import LightingDataModuleSubset
from embeddings.data.io import T_path
from embeddings.data.qa_datamodule import QAHuggingFaceDataModule
from embeddings.evaluator.evaluation_results import Predictions, QuestionAnsweringEvaluationResults
from embeddings.evaluator.question_answering_evaluator import QASquadV2Evaluator
from embeddings.model.lightning_model import LightningModel
from embeddings.pipeline.lightning_pipeline import LightningPipeline, EvaluationResult
from embeddings.task.lightning_task import question_answering
from embeddings.utils.loggers import LightningLoggingConfig
from embeddings.utils.utils import standardize_name


class LightningQuestionAnsweringPipeline(
    LightningPipeline[datasets.DatasetDict, Predictions, QuestionAnsweringEvaluationResults]
):
    def __init__(
        self,
        embedding_name_or_path: T_path,
        dataset_name_or_path: T_path,
        output_path: T_path,
        evaluation_filename: str = "evaluation.json",
        config: LightningConfig = LightningBasicConfig(),
        devices: Optional[Union[List[int], str, int]] = "auto",
        accelerator: Optional[Union[str, Accelerator]] = "auto",
        logging_config: LightningLoggingConfig = LightningLoggingConfig(),
        tokenizer_name_or_path: Optional[T_path] = None,
        predict_subset: LightingDataModuleSubset = LightingDataModuleSubset.TEST,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        model_checkpoint_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        task_train_kwargs = config.task_train_kwargs
        task_train_kwargs.update({"devices": devices, "accelerator": accelerator})
        tokenizer_name_or_path = tokenizer_name_or_path or embedding_name_or_path
        model_checkpoint_kwargs = model_checkpoint_kwargs if model_checkpoint_kwargs else {}
        output_path = Path(output_path)
        self.evaluation_filename = evaluation_filename
        datamodule = QAHuggingFaceDataModule(
            dataset_name_or_path=dataset_name_or_path,
            tokenizer_name_or_path=tokenizer_name_or_path,
            train_batch_size=config.task_model_kwargs["train_batch_size"],
            eval_batch_size=config.task_model_kwargs["eval_batch_size"],
            max_seq_length=config.task_model_kwargs["max_seq_length"]
            if config.task_model_kwargs["max_seq_length"] is not None
            else 512,
            doc_stride=config.task_model_kwargs["doc_stride"]
            if config.task_model_kwargs["doc_stride"] is not None
            else 128,
        )
        datamodule.setup(stage="fit")

        task = question_answering.QuestionAnsweringTask(
            model_name_or_path=tokenizer_name_or_path,
            finetune_last_n_layers=config.finetune_last_n_layers,
            task_model_kwargs=config.task_model_kwargs,
            output_path=".",
            model_config_kwargs=config.model_config_kwargs,
            task_train_kwargs=task_train_kwargs,
            early_stopping_kwargs=config.early_stopping_kwargs,
            model_checkpoint_kwargs=model_checkpoint_kwargs,
            mlflow_logger=None,
        )
        model = LightningModel(task, predict_subset)
        evaluator = QASquadV2Evaluator()
        super().__init__(datamodule, model, evaluator, output_path, logging_config)
