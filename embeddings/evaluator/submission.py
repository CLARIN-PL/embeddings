import json
import tempfile
from abc import ABC
from dataclasses import asdict, dataclass
from io import TextIOWrapper
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import srsly
import yaml
from wandb.apis.public import Run

from embeddings.data.io import T_path
from embeddings.evaluator.evaluation_results import Predictions
from embeddings.evaluator.sequence_labeling_evaluator import SequenceLabelingEvaluator
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.utils.json_dict_persister import CustomJsonEncoder
from embeddings.utils.utils import compress_and_remove, standardize_name


@dataclass
class _BaseSubmission(ABC):
    submission_name: str
    dataset_name: str
    dataset_version: str
    embedding_name: str
    hparams: Dict[str, Any]
    packages: List[str]
    predictions: Union[Predictions, List[Predictions]]
    config: Optional[Dict[str, Any]]  # any additional config

    def __post_init__(self) -> None:
        self.submission_name = standardize_name(self.submission_name)

    def save_json(
        self,
        root: T_path = ".",
        filename: Optional[str] = None,
        pred_filename: Optional[str] = None,
        compress: bool = True,
    ) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        filename = filename if filename else f"{self.submission_name}.json"
        with open(root.joinpath(filename), "w") as f:
            json.dump(self.without_predictions(), f, cls=CustomJsonEncoder, indent=2)

        pred_filename = (
            pred_filename if pred_filename else f"{self.submission_name}_predictions.json"
        )
        with open(root.joinpath(pred_filename), "w") as f:
            json.dump(self.predictions, f, cls=CustomJsonEncoder, indent=2)
        if compress:
            compress_and_remove(root.joinpath(filename))
            compress_and_remove(root.joinpath(pred_filename))

    def without_predictions(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("predictions")
        return result


@dataclass
class Submission(_BaseSubmission):
    predictions: Predictions
    metrics: Dict[str, Any]

    @staticmethod
    def _get_evaluator_kwargs(hparams: Dict[str, Any]) -> Dict[str, Any]:
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

    @classmethod
    def from_local_disk(
        cls,
        submission_name: str,
        evaluation_file_path: T_path,
        packages_file_path: T_path,
        wandb_config_path: T_path,
        best_params_path: T_path,
        task: str,
    ) -> "Submission":
        wandb_config_path = Path(wandb_config_path)
        if wandb_config_path.is_dir():
            [wandb_run_dir] = list(wandb_config_path.glob("run*"))
            wandb_config_path = Path(wandb_run_dir) / "files" / "config.yaml"

        with wandb_config_path.open() as f:
            wandb_cfg = yaml.load(f, Loader=yaml.Loader)
        with Path(evaluation_file_path).open() as f:
            evaluation_json = f.read()
        with Path(best_params_path).open() as f:
            hparams = yaml.load(f, Loader=yaml.Loader)

        evaluator_kwargs = cls._get_evaluator_kwargs(hparams)
        predictions = Predictions.from_evaluation_json(evaluation_json)
        evaluator = cls._get_evaluator_cls(task)(return_input_data=False, **evaluator_kwargs)
        metrics = evaluator.evaluate(data=predictions).metrics
        packages = srsly.read_json(str(packages_file_path))
        return cls(
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

    @staticmethod
    def from_wandb(
        submission_name: str,
        retrain_run: Run,
        hps_summary_run: Run,
        task: str,
        root: Optional[T_path] = None,
    ) -> "Submission":
        assert retrain_run.state == "finished"

        if root is None:
            tmp_dir = tempfile.TemporaryDirectory()
            root = tmp_dir.name
        else:
            tmp_dir = None

        # type ignore due to untyped files() function
        files = {
            file.name: file.download(root=Path(root).joinpath(retrain_run.name))
            for file in retrain_run.files(  # type: ignore[no-untyped-call]
                ["evaluation.json", "packages.json"]
            )
        }
        # type ignore due to untyped logged_artifacts() function
        [hps_output_artifact] = [
            artifact
            for artifact in hps_summary_run.logged_artifacts()  # type: ignore[no-untyped-call]
            if artifact.name.split(":")[0] == "hps_result"
        ]
        files["best_params.yaml"] = hps_output_artifact.download(
            root=Path(root).joinpath(hps_summary_run.name)
        ).joinpath("best_params.yaml")

        config = retrain_run.config
        with open(files["best_params.yaml"]) as f:
            hparams = yaml.load(f, Loader=yaml.Loader)
        evaluator_kwargs = Submission._get_evaluator_kwargs(hparams)
        predictions = Predictions.from_evaluation_json(files["evaluation.json"].read())
        evaluator = Submission._get_evaluator_cls(task)(return_input_data=False, **evaluator_kwargs)
        metrics = evaluator.evaluate(data=predictions).metrics

        packages = srsly.json_loads(files["packages.json"].read())

        for file in files.values():
            if isinstance(file, TextIOWrapper):
                file.close()

        if tmp_dir:
            tmp_dir.cleanup()

        return Submission(
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


@dataclass
class AveragedSubmission(_BaseSubmission):
    metrics: List[Dict[str, Any]]
    metrics_avg: Dict[str, Any]
    metrics_median: Dict[str, Any]
    metrics_std: Dict[str, Any]
    predictions: List[Predictions]
    averaged_over: int

    @classmethod
    def from_local_disk(
        cls,
        submission_name: str,
        evaluation_file_paths: List[T_path],
        packages_file_paths: List[T_path],
        wandb_config_paths: List[T_path],
        best_params_path: T_path,
        task: str,
    ) -> "AveragedSubmission":
        assert len(evaluation_file_paths) == len(packages_file_paths) == len(wandb_config_paths)
        submissions = [
            Submission.from_local_disk(
                submission_name,
                evaluation_file_path,
                packages_file_path,
                wandb_config_path,
                best_params_path,
                task,
            )
            for evaluation_file_path, packages_file_path, wandb_config_path in zip(
                evaluation_file_paths, packages_file_paths, wandb_config_paths
            )
        ]
        return cls.from_submissions(submissions)

    @classmethod
    def from_wandb(
        cls,
        submission_name: str,
        retrain_runs: Sequence[Run],
        hps_summary_run: Run,
        task: str,
        root: Optional[T_path] = None,
    ) -> "AveragedSubmission":
        submissions = [
            Submission.from_wandb(submission_name, run, hps_summary_run, task, root)
            for run in retrain_runs
        ]
        return cls.from_submissions(submissions)

    @classmethod
    def from_submissions(cls, submissions: Sequence[Submission]) -> "AveragedSubmission":
        cls._check_equal_submissions_dicts([asdict(submission) for submission in submissions])
        metrics = [submission.metrics for submission in submissions]
        metrics_avg, metrics_median, metrics_std = cls._aggregate_metrics_dicts(metrics)
        predictions = [submission.predictions for submission in submissions]
        return AveragedSubmission(
            predictions=predictions,
            metrics=metrics,
            metrics_avg=metrics_avg,
            metrics_median=metrics_median,
            metrics_std=metrics_std,
            averaged_over=len(submissions),
            **cls._get_common_fields(submissions[0]),
        )

    @classmethod
    def _aggregate_metrics_dicts(
        cls, dicts: Sequence[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        avg_dict, median_dict, std_dict = dict(), dict(), dict()
        for k in dicts[0].keys():
            if k in {"support"}:
                if not all(d[k] == dicts[0][k] for d in dicts):
                    raise ValueError(f"During aggregating dict values of key '{k}' were not equal.")
                avg_dict[k] = dicts[0][k]
            elif isinstance(dicts[0][k], (int, float)):
                avg_dict[k] = mean(d[k] for d in dicts)
                median_dict[k] = median(d[k] for d in dicts)
                std_dict[k] = stdev(d[k] for d in dicts)
            elif isinstance(dicts[0][k], dict):
                avg_dict[k], median_dict[k], std_dict[k] = cls._aggregate_metrics_dicts(
                    [d[k] for d in dicts if k in d]
                )
        return avg_dict, median_dict, std_dict

    @classmethod
    def _check_equal_submissions_dicts(cls, dicts: Sequence[Dict[str, Any]]) -> None:
        for k in dicts[0].keys():
            if k in {"metrics", "y_pred", "y_probabilities", "output_path", "start_time"}:
                continue
            elif isinstance(dicts[0][k], dict):
                cls._check_equal_submissions_dicts([d[k] for d in dicts])
            elif isinstance(dicts[0][k], np.ndarray):
                if not all((d[k] == dicts[0][k]).all() for d in dicts):
                    raise ValueError(f"Fields '{k}' of submissions are not equal.")
            elif not all(d[k] == dicts[0][k] for d in dicts):
                raise ValueError(f"Fields '{k}' of submissions are not equal.")

    @staticmethod
    def _get_common_fields(submission: Submission) -> Dict[str, Any]:
        result = asdict(submission)
        result.pop("predictions")
        result.pop("metrics")
        return result
