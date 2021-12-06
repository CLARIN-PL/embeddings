import abc
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl

from embeddings.data.huggingface_datamodule import TextClassificationDataModule
from embeddings.data.io import T_path
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.lightning_model import LightningModel
from embeddings.task.lightning_task.text_classification import TextClassificationTask


class LightningPipeline(abc.ABC):
    @abc.abstractmethod
    def run(self) -> Dict[str, Any]:
        pass


class LightningClassificationPipeline(LightningPipeline):
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
        self.datamodule = TextClassificationDataModule(
            model_name_or_path=embedding_name,
            dataset_name=dataset_name,
            text_fields=input_column_name,
            target_field=target_column_name,
            load_dataset_kwargs=load_dataset_kwargs,
        )
        self.datamodule.initalize()
        trainer = pl.Trainer(
            default_root_dir=output_path, **task_train_kwargs if task_train_kwargs else {}
        )
        task = TextClassificationTask(
            model_name_or_path=embedding_name,
            num_labels=self.datamodule.num_labels,
            **task_model_kwargs if task_model_kwargs else {},
        )
        self.model = LightningModel(trainer=trainer, task=task, predict_subset="test")
        self.evaluator = TextClassificationEvaluator()

    def run(self) -> Dict[str, Any]:
        self.datamodule.setup("fit")
        model_result = self.model.execute(data=self.datamodule)
        metrics = self.evaluator.evaluate(model_result)
        return metrics
