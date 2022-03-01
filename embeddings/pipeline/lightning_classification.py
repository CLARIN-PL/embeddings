from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import datasets
from numpy import typing as nptyping

from embeddings.config.lightning_config_space import LightningBasicConfigSpace, LightningConfigSpace
from embeddings.data.datamodule import TextClassificationDataModule
from embeddings.data.dataset import LightingDataModuleSubset
from embeddings.data.io import T_path
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.lightning_model import LightningModel
from embeddings.pipeline.lightning_pipeline import LightningPipeline
from embeddings.task.lightning_task.text_classification import TextClassificationTask
from embeddings.utils.json_dict_persister import JsonPersister
from embeddings.utils.loggers import LightningLoggingConfig
from embeddings.utils.utils import initialize_kwargs


class LightningClassificationPipeline(
    LightningPipeline[datasets.DatasetDict, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]]
):
    def __init__(
        self,
        embedding_name_or_path: T_path,
        dataset_name_or_path: T_path,
        input_column_name: Union[str, Sequence[str]],
        target_column_name: str,
        output_path: T_path,
        evaluation_filename: str = "evaluation.json",
        config_space: LightningConfigSpace = LightningBasicConfigSpace(),
        logging_config: LightningLoggingConfig = LightningLoggingConfig(),
        tokenizer_name_or_path: Optional[T_path] = None,
        predict_subset: LightingDataModuleSubset = LightingDataModuleSubset.TEST,
    ):
        tokenizer_name_or_path = tokenizer_name_or_path or embedding_name_or_path
        output_path = Path(output_path)
        self.evaluation_filename = evaluation_filename

        datamodule = TextClassificationDataModule(
            tokenizer_name_or_path=tokenizer_name_or_path,
            dataset_name_or_path=dataset_name_or_path,
            text_fields=input_column_name,
            target_field=target_column_name,
            train_batch_size=config_space.train_batch_size,
            eval_batch_size=config_space.eval_batch_size,
            tokenizer_kwargs=config_space.tokenizer_kwargs,
            batch_encoding_kwargs=config_space.batch_encoding_kwargs,
            load_dataset_kwargs=config_space.load_dataset_kwargs,
            **config_space.datamodule_kwargs
        )
        task = TextClassificationTask(
            model_name_or_path=embedding_name_or_path,
            output_path=output_path,
            finetune_last_n_layers=config_space.finetune_last_n_layers,
            model_config_kwargs=config_space.model_config_kwargs,
            task_model_kwargs=config_space.task_model_kwargs,
            task_train_kwargs=config_space.task_train_kwargs,
            early_stopping_kwargs=config_space.early_stopping_kwargs,
            logging_config=logging_config,
        )
        model = LightningModel(task=task, predict_subset=predict_subset)
        evaluator = TextClassificationEvaluator().persisting(
            JsonPersister(path=output_path.joinpath(evaluation_filename))
        )
        super().__init__(datamodule, model, evaluator, output_path, logging_config)
