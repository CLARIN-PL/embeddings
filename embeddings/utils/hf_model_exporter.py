import dataclasses
import pathlib
from typing import Any, Dict, List, Optional, TypeVar

import srsly
from transformers import AutoModel, AutoTokenizer

from embeddings.data.datamodule import Data
from embeddings.data.io import T_path
from embeddings.model.lightning_module.lightning_module import LightningModule, Model
from embeddings.pipeline.lightning_pipeline import EvaluationResult, LightningPipeline, ModelResult
from embeddings.task.lightning_task.lightning_task import LightningTask
from embeddings.utils.utils import get_installed_packages

TaskModel = TypeVar("TaskModel", bound=LightningTask)


@dataclasses.dataclass
class HuggingFaceModelExporter:
    path: T_path
    add_installed_packages_file: bool = True
    add_hparams_configuration: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.path, pathlib.Path):
            self.path = pathlib.Path(self.path)

    @staticmethod
    def _check_model(model: Optional[LightningModule[Model]]) -> None:
        if not model:
            raise LightningTask.MODEL_UNDEFINED_EXCEPTION

    @staticmethod
    def _check_tokenizer(tokenizer: Optional[AutoTokenizer], msg: str = "") -> None:
        if not tokenizer:
            raise ValueError(f"Tokenizer not found! {msg}")

    @staticmethod
    def _map_target_names(model: AutoModel, target_names: List[str]) -> AutoModel:
        if target_names:
            id2label = {k: v for k, v in zip(model.config.id2label.keys(), target_names)}
            label2id = {k: v for k, v in zip(target_names, model.config.id2label.keys())}

            model.config.id2label = id2label
            model.config.label2id = label2id

        return model

    def export_model_from_pipeline(
        self,
        pipeline: LightningPipeline[Data, ModelResult, EvaluationResult],
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        task = getattr(pipeline.model, "task")
        assert task

        self.export_model_from_task(task=task, tokenizer=tokenizer)

    def export_model_from_task(
        self, task: TaskModel, tokenizer: Optional[AutoTokenizer] = None
    ) -> None:
        HuggingFaceModelExporter._check_model(task.model)
        if not tokenizer:
            HuggingFaceModelExporter._check_tokenizer(
                task.tokenizer,
                msg="Tokenizer not found, re-run pipeline/task or pass tokenizer explicitly!",
            )
            tokenizer = task.tokenizer

        model = getattr(task.model, "model", None)
        hparams = getattr(task.model, "hparams", None)
        assert model
        assert hparams

        self.export(
            model=model,
            tokenizer=tokenizer,
            target_names=getattr(task, "target_names", None),
            hparams=hparams if self.add_hparams_configuration else None,
        )

    def export(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        target_names: Optional[List[str]],
        hparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        assert isinstance(self.path, pathlib.Path)
        self.path.mkdir(parents=True, exist_ok=False)

        if self.add_hparams_configuration:
            if hparams:
                srsly.write_json(self.path / "hparams.json", hparams)
            else:
                raise ValueError(
                    "hparams are not found, set `add_hparams_configuration` to False or pass `hparams` dict"
                )

        if self.add_installed_packages_file:
            srsly.write_json(self.path / "packages.json", get_installed_packages())

        if target_names:
            model = HuggingFaceModelExporter._map_target_names(
                model=model, target_names=target_names
            )

        model.save_pretrained(self.path)
        tokenizer.save_pretrained(self.path)
