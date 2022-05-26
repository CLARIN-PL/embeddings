from pathlib import Path
from typing import List

import typer

from embeddings.data.io import T_path
from embeddings.defaults import SUBMISSIONS_PATH
from embeddings.evaluator.evaluation_results import Task
from embeddings.evaluator.submission import AveragedSubmission, Submission

app = typer.Typer()


def run(
    dataset: str = typer.Option(..., help="Dataset"),
    model: str = typer.Option(..., help="Model"),
    task: Task = typer.Option(..., help="Task type"),
    evaluation_file_paths: List[Path] = typer.Option(
        ..., dir_okay=False, file_okay=True, exists=True, help="Evaluation file path"
    ),
    packages_file_paths: List[Path] = typer.Option(
        ..., dir_okay=False, file_okay=True, exists=True, help="Packages file path"
    ),
    wandb_log_dirs: List[Path] = typer.Option(
        ..., dir_okay=True, file_okay=False, exists=True, help="Wandb log dir"
    ),
    best_params_path: Path = typer.Option(..., help="Best parameters path"),
) -> None:
    # assert Path(wandb_log_dir).is_dir()
    submission_name = f"{dataset}_{model}"

    evaluation_file_paths_: List[T_path] = list(evaluation_file_paths)
    packages_file_paths_: List[T_path] = list(packages_file_paths)
    wandb_log_dirs_: List[T_path] = list(wandb_log_dirs)

    submission = AveragedSubmission.from_local_disk(
        submission_name=submission_name,
        task=task,
        evaluation_file_paths=evaluation_file_paths_,
        packages_file_paths=packages_file_paths_,
        wandb_log_dirs=wandb_log_dirs_,
        best_params_path=best_params_path,
    )
    submission.save_json(SUBMISSIONS_PATH / submission_name)


typer.run(run)
