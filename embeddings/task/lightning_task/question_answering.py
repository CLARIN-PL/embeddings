import os
import warnings
from typing import Any, Dict, Optional, Union

from embeddings.data.datamodule import HuggingFaceDataModule
from embeddings.data.dataset import LightingDataModuleSubset
from embeddings.data.io import T_path
from embeddings.model.lightning_module.question_answering import QuestionAnsweringModule
from embeddings.task.lightning_task.lightning_task import LightningTask
from embeddings.utils.loggers import LightningLoggingConfig


class QuestionAnsweringTask(LightningTask):
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
            logging_config=LightningLoggingConfig.from_flags(),
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
        )

    def predict(self, dataloader: Any, return_names: bool = True) -> Any:
        assert self.model is not None
        assert self.trainer is not None
        return self.trainer.predict(model=self.model, dataloaders=dataloader)

    def postprocess(
        self, data: HuggingFaceDataModule, predict_subset: Union[str, LightingDataModuleSubset]
    ) -> Dict[str, Any]:
        data_loader = data.get_subset(predict_subset)
        scores = {}
        model_outputs = self.predict(data_loader)
        scores[predict_subset] = {
            "examples": data.dataset_raw[predict_subset].to_pandas(),
            "outputs": model_outputs,
            "overflow_to_sample_mapping": data.overflow_to_sample_mapping[predict_subset],
            "offset_mapping": data.offset_mapping[predict_subset],
        }
        return scores

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: T_path,
        output_path: T_path,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
        logging_config: Optional[LightningLoggingConfig] = None,
    ) -> "LightningTask":
        return cls.restore_task_model(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            task_train_kwargs=task_train_kwargs,
            lightning_module=QuestionAnsweringModule,
            logging_config=logging_config,
        )
