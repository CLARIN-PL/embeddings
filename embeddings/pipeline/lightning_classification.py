from typing import Any, Dict, List, Optional, Union

import datasets
import numpy as np
import pytorch_lightning as pl

from embeddings.data.datamodule import TextClassificationDataModule
from embeddings.data.io import T_path
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.lightning_model import LightningModel
from embeddings.pipeline.lightning_pipeline import LightningPipeline
from embeddings.task.lightning_task.text_classification import TextClassificationTask


class LightningClassificationPipeline(
    LightningPipeline[datasets.DatasetDict, Dict[str, np.ndarray], Dict[str, Any]]
):
    def __init__(
        self,
        embedding_name: str,
        dataset_name: str,
        input_column_name: Union[str, List[str]],
        target_column_name: str,
        output_path: T_path,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
    ):
        datamodule = TextClassificationDataModule(
            model_name_or_path=embedding_name,
            dataset_name=dataset_name,
            text_fields=input_column_name,
            target_field=target_column_name,
            load_dataset_kwargs=load_dataset_kwargs,
        )
        datamodule.initalize()
        trainer = pl.Trainer(
            default_root_dir=output_path, **task_train_kwargs if task_train_kwargs else {}
        )
        task = TextClassificationTask(
            model_name_or_path=embedding_name,
            num_labels=datamodule.num_labels,
            task_model_kwargs=task_model_kwargs,
        )
        model = LightningModel(trainer=trainer, task=task, predict_subset="test")
        evaluator = TextClassificationEvaluator()
        super().__init__(datamodule, model, evaluator)
