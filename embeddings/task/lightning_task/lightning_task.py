import abc
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Sequence, Type, TypeVar, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from transformers import AutoModel

from embeddings.data.datamodule import HuggingFaceDataModule
from embeddings.data.dataset import LightingDataModuleSubset
from embeddings.data.io import T_path
from embeddings.data.qa_datamodule import QuestionAnsweringDataModule
from embeddings.evaluator.evaluation_results import Predictions
from embeddings.model.lightning_module.huggingface_module import HuggingFaceLightningModule
from embeddings.model.lightning_module.lightning_module import LightningModule
from embeddings.task.task import Output, Task
from embeddings.utils.lightning_callbacks.best_epoch_callback import BestEpochCallback
from embeddings.utils.loggers import LightningLoggingConfig, get_logger
from embeddings.utils.torch_utils import cleanup_torch_model_artifacts

_logger = get_logger(__name__)

LightningDataModules = Union[HuggingFaceDataModule, QuestionAnsweringDataModule]
LightningDataModule = TypeVar("LightningDataModule", bound=LightningDataModules)


class LightningTask(Task[LightningDataModule, Output], Generic[LightningDataModule, Output]):
    MODEL_UNDEFINED_EXCEPTION = ValueError("Model undefined. Use build_task_model() first!")

    def __init__(
        self,
        output_path: T_path,
        task_train_kwargs: Dict[str, Any],
        early_stopping_kwargs: Dict[str, Any],
        model_checkpoint_kwargs: Dict[str, Any],
        logging_config: LightningLoggingConfig,
    ):
        super().__init__()
        self.output_path = Path(output_path)
        self.task_train_kwargs = task_train_kwargs
        self.early_stopping_kwargs = early_stopping_kwargs
        self.model_checkpoint_kwargs = model_checkpoint_kwargs
        self.model: Optional[HuggingFaceLightningModule] = None
        self.trainer: Optional[pl.Trainer] = None
        self.logging_config = logging_config

    @property
    def best_epoch(self) -> Optional[float]:
        if self.trainer is None:
            return None

        for callback in self.trainer.callbacks:
            if isinstance(callback, BestEpochCallback):
                return callback.best_epoch
        return None

    @property
    def best_validation_score(self) -> Optional[float]:
        if self.trainer is None:
            return None

        for callback in self.trainer.callbacks:
            if isinstance(callback, BestEpochCallback):
                return callback.best_score.item()
        return None

    def _get_callbacks(self, dataset_subsets: Sequence[str]) -> List[Callback]:
        callbacks: List[Callback] = [
            ModelCheckpoint(
                dirpath=self.output_path.joinpath("checkpoints"), **self.model_checkpoint_kwargs
            )
        ]
        if "validation" in dataset_subsets:
            callbacks.append(BestEpochCallback())
            if self.early_stopping_kwargs:
                callbacks.append(EarlyStopping(**self.early_stopping_kwargs))
        return callbacks

    def fit(
        self,
        data: LightningDataModule,
        run_name: Optional[str] = None,
    ) -> None:
        if not self.model:
            raise self.MODEL_UNDEFINED_EXCEPTION

        callbacks = self._get_callbacks(dataset_subsets=list(data.load_dataset().keys()))
        self.trainer = pl.Trainer(
            default_root_dir=str(self.output_path),
            callbacks=callbacks,
            logger=self.logging_config.get_lightning_loggers(self.output_path, run_name),
            **self.task_train_kwargs
        )
        try:
            self.trainer.fit(self.model, data)
        except Exception as e:
            del self.trainer
            cleanup_torch_model_artifacts()
            raise e

    @abc.abstractmethod
    def fit_predict(
        self,
        data: LightningDataModule,
        predict_subset: LightingDataModuleSubset = LightingDataModuleSubset.TEST,
        run_name: Optional[str] = None,
    ) -> Output:
        pass

    @abc.abstractmethod
    def predict(self, dataloader: DataLoader[Any], return_names: bool = True) -> Output:
        pass

    @abc.abstractmethod
    def build_task_model(self) -> None:
        pass

    @classmethod
    @abc.abstractmethod
    def from_checkpoint(
        cls,
        checkpoint_path: T_path,
        output_path: T_path,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
        logging_config: Optional[LightningLoggingConfig] = None,
    ) -> "LightningTask[LightningDataModule, Output]":
        pass

    @classmethod
    @abc.abstractmethod
    def restore_task_model(
        cls,
        checkpoint_path: T_path,
        output_path: T_path,
        lightning_module: Type[LightningModule[AutoModel]],
        task_train_kwargs: Optional[Dict[str, Any]],
        logging_config: Optional[LightningLoggingConfig],
    ) -> "LightningTask[LightningDataModule, Output]":
        pass


class ClassificationLightningTask(LightningTask[HuggingFaceDataModule, Predictions]):
    def __init__(
        self,
        output_path: T_path,
        task_train_kwargs: Dict[str, Any],
        early_stopping_kwargs: Dict[str, Any],
        model_checkpoint_kwargs: Dict[str, Any],
        logging_config: LightningLoggingConfig,
    ):
        super().__init__(
            output_path=output_path,
            task_train_kwargs=task_train_kwargs,
            early_stopping_kwargs=early_stopping_kwargs,
            model_checkpoint_kwargs=model_checkpoint_kwargs,
            logging_config=logging_config,
        )

    def fit_predict(
        self,
        data: HuggingFaceDataModule,
        predict_subset: LightingDataModuleSubset = LightingDataModuleSubset.TEST,
        run_name: Optional[str] = None,
    ) -> Predictions:
        if not self.model:
            raise self.MODEL_UNDEFINED_EXCEPTION
        self.fit(data, run_name=run_name)
        dataloader = data.get_subset(subset=predict_subset)
        assert isinstance(dataloader, DataLoader)
        result = self.predict(dataloader=dataloader)
        return result

    @classmethod
    def restore_task_model(
        cls,
        checkpoint_path: T_path,
        output_path: T_path,
        lightning_module: Type[LightningModule[AutoModel]],
        task_train_kwargs: Optional[Dict[str, Any]],
        logging_config: Optional[LightningLoggingConfig],
    ) -> "ClassificationLightningTask":
        model = lightning_module.load_from_checkpoint(str(checkpoint_path))
        trainer = pl.Trainer(default_root_dir=str(output_path), **task_train_kwargs or {})
        init_kwargs = {
            "model_name_or_path": model.hparams.model_name_or_path,
            "output_path": output_path,
            "num_classes": model.hparams.num_classes,
            "finetune_last_n_layers": model.hparams.finetune_last_n_layers,
            "model_config_kwargs": model.hparams.config_kwargs,
            "task_model_kwargs": model.hparams.task_model_kwargs,
            "task_train_kwargs": task_train_kwargs or {},
            "early_stopping_kwargs": {},
            "model_checkpoint_kwargs": {},
            "logging_config": logging_config or LightningLoggingConfig(),
        }
        task = cls(**init_kwargs)
        task.model = model
        task.trainer = trainer
        model.trainer = trainer
        return task
