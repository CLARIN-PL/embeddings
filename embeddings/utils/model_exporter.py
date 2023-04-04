import abc
import dataclasses
import json
import pathlib
from typing import Any, Dict, List, Optional

import srsly
import transformers.onnx
from datasets import ClassLabel
from transformers import AutoModel, AutoTokenizer
from transformers.onnx import FeaturesManager

from embeddings.data.datamodule import Data
from embeddings.data.io import T_path
from embeddings.pipeline.lightning_pipeline import EvaluationResult, LightningPipeline, ModelResult
from embeddings.task.lightning_task import SUPPORTED_HF_TASKS
from embeddings.task.lightning_task.lightning_task import LightningTask
from embeddings.utils.loggers import get_logger
from embeddings.utils.utils import get_installed_packages

_logger = get_logger(__name__)


class ExportMisconfigurationError(Exception):
    pass


@dataclasses.dataclass
class BaseModelExporter(abc.ABC):
    path: dataclasses.InitVar[T_path]
    add_installed_packages_file: bool = True
    add_hparams_configuration: bool = True

    _export_path: pathlib.Path = dataclasses.field(init=False)
    _task_to_export: SUPPORTED_HF_TASKS = dataclasses.field(init=False)
    _tokenizer_to_export: AutoTokenizer = dataclasses.field(init=False)
    _model_to_export: AutoModel = dataclasses.field(init=False)

    def __post_init__(self, path: T_path) -> None:
        self._export_path = pathlib.Path(path) if not isinstance(path, pathlib.Path) else path
        self._export_path.mkdir(parents=True, exist_ok=False)

    @staticmethod
    def _check_tokenizer(tokenizer: Optional[AutoTokenizer]) -> None:
        if not tokenizer:
            raise ExportMisconfigurationError(
                "Tokenizer not found, re-run pipeline/task or pass tokenizer explicitly!"
            )

    def _check_task(self) -> None:
        model = getattr(self._task_to_export.model, "model", None)
        if not model:
            raise LightningTask.MODEL_UNDEFINED_EXCEPTION

    @staticmethod
    def _map_target_names(model: AutoModel, target_names: List[str]) -> AutoModel:
        if target_names:
            assert len(target_names) == len(model.config.id2label.keys())
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
        self,
        task: SUPPORTED_HF_TASKS,
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        self._task_to_export = task
        self._check_task()

        model: AutoModel = getattr(self._task_to_export.model, "model", None)
        assert model
        self._model_to_export = model

        if not tokenizer:
            _logger.warning("Tokenizer not passed! Trying to use task tokenizer.")
            BaseModelExporter._check_tokenizer(task.tokenizer)
            _logger.warning("Tokenizer found, using task tokenizer.")
            tokenizer = task.tokenizer
        self._tokenizer_to_export = tokenizer

        label_names = None
        if task.hf_task_name.value == "sequence-classification":
            label_names = getattr(task, "target_names", None)
        elif task.hf_task_name.value == "token-classification":
            class_label: Optional[ClassLabel] = getattr(
                self._task_to_export.model, "class_label", None
            )
            if class_label:
                label_names = class_label.names

        if label_names:
            self._model_to_export = BaseModelExporter._map_target_names(
                self._model_to_export, target_names=label_names
            )

        hparams = getattr(self._task_to_export.model, "hparams", None)
        if self.add_hparams_configuration and not hparams:
            raise ExportMisconfigurationError(
                "Couldn't obtain hparams from task. "
                'Check your task model or set "add_hparams_configuration" parameter to "False"'
            )

        self._export_task_artifacts(hparams=hparams)
        self._export_model()

    def _export_task_artifacts(
        self,
        hparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.add_hparams_configuration:
            srsly.write_json(self._export_path / "hparams.json", hparams)

        if self.add_installed_packages_file:
            srsly.write_json(self._export_path / "packages.json", get_installed_packages())

    @abc.abstractmethod
    def _export_model(self) -> None:
        pass


@dataclasses.dataclass
class HuggingFaceModelExporter(BaseModelExporter):
    def _export_model(self) -> None:
        self._model_to_export.save_pretrained(self._export_path)
        self._tokenizer_to_export.save_pretrained(self._export_path)


class ONNXModelExporter(BaseModelExporter):
    _onnx_export_metadata: Dict[str, Any] = dataclasses.field(init=False)

    @property
    def onnx_export_metadata(self) -> Dict[str, Any]:
        if self._onnx_export_metadata:
            return self._onnx_export_metadata
        raise ValueError('"Onnx_export_metadata" is unset! Have you forget to export model first?')

    def _export_model(self) -> None:
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
            model=self._model_to_export, feature=self._task_to_export.hf_task_name.value
        )
        onnx_config = model_onnx_config(self._model_to_export.config)
        onnx_inputs, onnx_outputs = transformers.onnx.export(
            preprocessor=self._tokenizer_to_export,
            model=self._model_to_export,
            config=onnx_config,
            output=self._export_path / "model.onnx",
            opset=onnx_config.default_onnx_opset,
        )
        self._tokenizer_to_export.save_pretrained(self._export_path)
        with open(self._export_path / "config.json", "w") as f:
            json.dump(obj=self._model_to_export.config.to_dict(), fp=f)

        self._onnx_export_metadata = {
            "onnx_config": onnx_config,
            "onnx_inputs": onnx_inputs,
            "onnx_outputs": onnx_outputs,
        }
