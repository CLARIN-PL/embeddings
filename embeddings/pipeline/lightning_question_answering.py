from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets
from pytorch_lightning.accelerators import Accelerator

from embeddings.config.lightning_config import LightningQABasicConfig, LightningQAConfig
from embeddings.data.dataset import LightingDataModuleSubset
from embeddings.data.io import T_path
from embeddings.data.qa_datamodule import QuestionAnsweringDataModule
from embeddings.evaluator.evaluation_results import QuestionAnsweringEvaluationResults
from embeddings.evaluator.question_answering_evaluator import QuestionAnsweringEvaluator
from embeddings.model.lightning_model import LightningModel
from embeddings.pipeline.lightning_pipeline import LightningPipeline
from embeddings.task.lightning_task import question_answering as qa
from embeddings.utils.loggers import LightningLoggingConfig


class LightningQuestionAnsweringPipeline(
    LightningPipeline[datasets.DatasetDict, Dict[str, Any], QuestionAnsweringEvaluationResults]
):
    def __init__(
        self,
        embedding_name_or_path: T_path,
        dataset_name_or_path: T_path,
        context_column_name: str,
        question_column_name: str,
        answer_column_name: str,
        output_path: T_path,
        evaluation_filename: str = "evaluation.json",
        config: LightningQAConfig = LightningQABasicConfig(),
        devices: Optional[Union[List[int], str, int]] = "auto",
        accelerator: Optional[Union[str, Accelerator]] = "auto",
        logging_config: LightningLoggingConfig = LightningLoggingConfig(),
        tokenizer_name_or_path: Optional[T_path] = None,
        predict_subset: LightingDataModuleSubset = LightingDataModuleSubset.TEST,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        model_checkpoint_kwargs: Optional[Dict[str, Any]] = None,
        compile_model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.evaluation_filename = evaluation_filename

        task_train_kwargs = config.task_train_kwargs
        task_train_kwargs.update({"devices": devices, "accelerator": accelerator})
        tokenizer_name_or_path = tokenizer_name_or_path or embedding_name_or_path
        model_checkpoint_kwargs = model_checkpoint_kwargs if model_checkpoint_kwargs else {}
        output_path = Path(output_path)
        pipeline_kwargs = locals()

        datamodule = QuestionAnsweringDataModule(
            dataset_name_or_path=dataset_name_or_path,
            context_field=context_column_name,
            question_field=question_column_name,
            target_field=answer_column_name,
            tokenizer_name_or_path=tokenizer_name_or_path,
            train_batch_size=config.task_model_kwargs["train_batch_size"],
            eval_batch_size=config.task_model_kwargs["eval_batch_size"],
            doc_stride=config.task_model_kwargs["doc_stride"],
            batch_encoding_kwargs=config.batch_encoding_kwargs,
            load_dataset_kwargs=load_dataset_kwargs,
            **config.datamodule_kwargs
        )
        datamodule.setup(stage="fit")

        task = qa.QuestionAnsweringTask(
            model_name_or_path=tokenizer_name_or_path,
            finetune_last_n_layers=config.finetune_last_n_layers,
            task_model_kwargs=config.task_model_kwargs,
            output_path=".",
            model_config_kwargs=config.model_config_kwargs,
            task_train_kwargs=task_train_kwargs,
            early_stopping_kwargs=config.early_stopping_kwargs,
            model_checkpoint_kwargs=model_checkpoint_kwargs,
            compile_model_kwargs=compile_model_kwargs,
        )
        model: LightningModel[QuestionAnsweringDataModule, Dict[str, Any]] = LightningModel(
            task, predict_subset
        )
        evaluator = QuestionAnsweringEvaluator()
        super().__init__(
            datamodule,
            model,
            evaluator,
            output_path,
            logging_config,
            pipeline_kwargs=pipeline_kwargs,
        )
