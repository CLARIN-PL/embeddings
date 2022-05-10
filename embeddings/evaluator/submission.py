import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

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
    metrics: Dict[str, Any]
    hparams: Dict[str, Any]
    predictions: Predictions
    packages: List[str]
    config: Optional[Dict[str, Any]] = None  # any additional config

    @staticmethod
    def _init_evaluator_kwargs(hparams: Dict[str, Any]) -> Dict[str, Any]:
        evaluator_kwargs = {}
        evaluation_mode = hparams.get("evaluation_mode", None)
        tagging_scheme = hparams.get("tagging_scheme", None)

        if evaluation_mode or tagging_scheme:
            evaluator_kwargs.update(
                {
                    "evaluation_mode": evaluation_mode,
                    "tagging_scheme": tagging_scheme,
                }
            )
        return evaluator_kwargs

    @staticmethod
    def from_local_disk(
        submission_name: str,
        evaluation_file_path: T_path,
        packages_file_path: T_path,
        wandb_log_dir: T_path,
        best_params_path: T_path,
        task: str,
    ) -> "Submission":
        [wandb_run_dir] = [it for it in Path(wandb_log_dir).iterdir() if "run" in str(it)]
        wandb_config_path = wandb_run_dir / "files" / "config.yaml"

        with wandb_config_path.open() as f:
            wandb_cfg = yaml.load(f, Loader=yaml.Loader)
        with Path(evaluation_file_path).open() as f:
            evaluation_json = f.read()
        with Path(best_params_path).open() as f:
            hparams = yaml.load(f, Loader=yaml.Loader)

        evaluator_kwargs = Submission._init_evaluator_kwargs(hparams)
        predictions = Predictions.from_evaluation_json(evaluation_json)
        evaluator = Submission._get_evaluator_cls(task)(return_input_data=False, **evaluator_kwargs)
        metrics = evaluator.evaluate(data=predictions).metrics
        packages = srsly.read_json(str(packages_file_path))
        submission = Submission(
            submission_name=submission_name,
            dataset_name=wandb_cfg["dataset_name_or_path"],
            dataset_version=wandb_cfg["dataset_version"],
            embedding_name=wandb_cfg["embedding_name_or_path"],
            metrics=metrics,
            predictions=predictions,
            hparams=hparams["config"],
            packages=packages,
            config=wandb_cfg,
        )
        return submission

    @staticmethod
    def from_wandb_run(
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
        hparams = yaml.load(files["best_params.yaml"].read(), Loader=yaml.Loader)
        evaluator_kwargs = Submission._init_evaluator_kwargs(hparams)
        predictions = Predictions.from_evaluation_json(files["evaluation.json"].read())
        evaluator = Submission._get_evaluator_cls(task)(return_input_data=False, **evaluator_kwargs)
        metrics = evaluator.evaluate(data=predictions).metrics

        packages = srsly.json_loads(files["packages.json"].read())
        submission = Submission(
            submission_name=submission_name,
            dataset_name=config["dataset_name_or_path"],
            dataset_version=config["dataset_version"],
            embedding_name=config["embedding_name_or_path"],
            metrics=metrics,
            predictions=predictions,
            hparams=hparams["config"],
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
