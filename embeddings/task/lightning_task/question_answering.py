import os
import warnings
from typing import Any, Dict, Optional, Type, Union

import datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoModel

from embeddings.data.dataset import LightingDataModuleSubset
from embeddings.data.io import T_path
from embeddings.data.qa_datamodule import QuestionAnsweringDataModule
from embeddings.model.lightning_module.lightning_module import LightningModule
from embeddings.model.lightning_module.question_answering import QuestionAnsweringModule
from embeddings.task.lightning_task.hf_task import HuggingFaceTaskName
from embeddings.task.lightning_task.lightning_task import LightningTask
from embeddings.utils.loggers import LightningLoggingConfig


class QuestionAnsweringTask(LightningTask[QuestionAnsweringDataModule, Dict[str, Any]]):
    def __init__(
        self,
        model_name_or_path: T_path,
        output_path: T_path,
        model_config_kwargs: Dict[str, Any],
        task_model_kwargs: Dict[str, Any],
        task_train_kwargs: Dict[str, Any],
        early_stopping_kwargs: Dict[str, Any],
        model_checkpoint_kwargs: Dict[str, Any],
        finetune_last_n_layers: int = -1,
        compile_model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
        # disable a thousand of warnings of HF
        warnings.simplefilter("ignore")
        # set os environ variable for multiprocesses
        os.environ["PYTHONWARNINGS"] = "ignore"

        super().__init__(
            output_path=output_path,
            task_train_kwargs=task_train_kwargs,
            early_stopping_kwargs=early_stopping_kwargs,
            model_checkpoint_kwargs=model_checkpoint_kwargs,
            compile_model_kwargs=compile_model_kwargs,
            logging_config=LightningLoggingConfig.from_flags(),
            hf_task_name=HuggingFaceTaskName.question_answering,
        )
        self.model_name_or_path = model_name_or_path
        self.model_config_kwargs = model_config_kwargs
        self.task_model_kwargs = task_model_kwargs
        self.finetune_last_n_layers = finetune_last_n_layers
        self.task_train_kwargs = task_train_kwargs

    def build_task_model(self) -> None:
        self.model = QuestionAnsweringModule(
            model_name_or_path=self.model_name_or_path,
            finetune_last_n_layers=self.finetune_last_n_layers,
            config_kwargs=self.model_config_kwargs,
            task_model_kwargs=self.task_model_kwargs,
            model_compile_kwargs=self.compile_model_kwargs,
        )

    def predict(self, dataloader: Any, return_names: bool = True) -> Any:
        assert self.model is not None
        assert self.trainer is not None
        return self.trainer.predict(model=self.model, dataloaders=dataloader)

    @staticmethod
    def postprocess_outputs(
        model_outputs: Any,
        data: QuestionAnsweringDataModule,
        predict_subset: Union[str, LightingDataModuleSubset],
    ) -> Dict[str, Any]:
        assert isinstance(data.dataset_raw, datasets.DatasetDict)
        return {
            "examples": data.dataset_raw[predict_subset].to_pandas(),
            "outputs": model_outputs,
            "overflow_to_sample_mapping": data.overflow_to_sample_mapping[predict_subset],
            "offset_mapping": data.offset_mapping[predict_subset],
        }

    def fit_predict(
        self,
        data: QuestionAnsweringDataModule,
        predict_subset: LightingDataModuleSubset = LightingDataModuleSubset.TEST,
        run_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.model:
            raise self.MODEL_UNDEFINED_EXCEPTION
        self.fit(data, run_name=run_name)

        dataloader = data.get_subset(subset=predict_subset)
        assert isinstance(dataloader, DataLoader)
        model_outputs = self.predict(dataloader=dataloader)
        result = self.postprocess_outputs(
            model_outputs=model_outputs, data=data, predict_subset=predict_subset
        )
        return result

    @classmethod
    def restore_task_model(
        cls,
        checkpoint_path: T_path,
        output_path: T_path,
        lightning_module: Type[LightningModule[AutoModel]],
        task_train_kwargs: Optional[Dict[str, Any]],
        logging_config: Optional[LightningLoggingConfig],
    ) -> "QuestionAnsweringTask":
        model = lightning_module.load_from_checkpoint(str(checkpoint_path))
        trainer = pl.Trainer(default_root_dir=str(output_path), **task_train_kwargs or {})
        hparams = getattr(model, "hparams")
        init_kwargs = {
            "model_name_or_path": hparams.model_name_or_path,
            "output_path": output_path,
            "finetune_last_n_layers": hparams.finetune_last_n_layers,
            "model_config_kwargs": hparams.config_kwargs,
            "task_model_kwargs": hparams.task_model_kwargs,
            "task_train_kwargs": task_train_kwargs or {},
            "early_stopping_kwargs": {},
            "model_checkpoint_kwargs": {},
            "logging_config": logging_config or LightningLoggingConfig(),
        }
        task = cls(**init_kwargs)
        task.model = model
        task.trainer = trainer
        # Due to "Self? has no attribute "trainer"" error
        model.trainer = trainer  # type: ignore[attr-defined]
        return task

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: T_path,
        output_path: T_path,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
        logging_config: Optional[LightningLoggingConfig] = None,
    ) -> "QuestionAnsweringTask":
        return cls.restore_task_model(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            task_train_kwargs=task_train_kwargs,
            lightning_module=QuestionAnsweringModule,
            logging_config=logging_config,
        )
