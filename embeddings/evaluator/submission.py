import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Type, Union

import srsly
import yaml
from wandb.apis.public import Run

from embeddings.data.io import T_path
from embeddings.evaluator.evaluation_results import Predictions
from embeddings.evaluator.sequence_labeling_evaluator import SequenceLabelingEvaluator
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.utils.json_dict_persister import CustomJsonEncoder


@dataclass
class Submission:
    submission_name: str
    dataset_name: str
    dataset_version: str
    embedding_name: str
    metrics: dict[str, Any]
    hparams: dict[str, Any]
    predictions: Predictions
    packages: list[str]
    config: Optional[dict[str, Any]] = None  # any additional config

    @staticmethod
    def from_run(
        submission_name: str, run: Run, task: str, root: Optional[T_path] = None
    ) -> "Submission":
        assert run.state == "finished"

        if root is None:
            tmp_dir = tempfile.TemporaryDirectory()
            root = tmp_dir.name
        else:
            tmp_dir = None

        files = {
            file.name: file.download(root=Path(root).joinpath(run.name))
            for file in run.files()
            if file.name in ["evaluation.json", "packages.json", "best_params.yaml"]
        }

        config = run.config
        predictions = Predictions.from_evaluation_json(files["evaluation.json"].read())
        evaluator = Submission._get_evaluator_cls(task)(return_input_data=False)
        metrics = evaluator.evaluate(data=predictions).metrics
        hparams = yaml.load(files["best_params.yaml"].read(), Loader=yaml.Loader)
        packages = srsly.json_loads(files["packages.json"].read())
        submission = Submission(
            submission_name=submission_name,
            dataset_name=config["dataset_name_or_path"],
            dataset_version=config["dataset_version"],
            embedding_name=config["embedding_name_or_path"],
            metrics=metrics,
            predictions=predictions,
            hparams=hparams,
            packages=packages,
            config=config,
        )

        for file in files.values():
            file.close()

        if tmp_dir:
            tmp_dir.cleanup()

        return submission

    @staticmethod
    def _get_evaluator_cls(
        task: str,
    ) -> Union[Type[TextClassificationEvaluator], Type[SequenceLabelingEvaluator]]:
        if task == "text_classification":
            return TextClassificationEvaluator
        elif task == "sequence_labeling":
            return SequenceLabelingEvaluator
        else:
            raise ValueError(f"Unrecognised task {task}.")

    def save_json(
        self, root: T_path = ".", filename: Optional[str] = None, compress: bool = True
    ) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        filename = filename if filename else f"{self.submission_name}.json"
        file_path = root.joinpath(filename)
        with open(file_path, "w") as f:
            json.dump(self, f, cls=CustomJsonEncoder, indent=2)
        if compress:
            with zipfile.ZipFile(
                root.joinpath(f"{filename}.zip"), mode="w", compression=zipfile.ZIP_DEFLATED
            ) as arc:
                arc.write(root.joinpath(filename), arcname=filename)
            root.joinpath(filename).unlink()
