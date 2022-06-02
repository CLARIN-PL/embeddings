from pathlib import Path

import typer

from embeddings.defaults import SUBMISSIONS_PATH
from embeddings.evaluator.evaluation_results import Task
from embeddings.evaluator.submission import Submission

app = typer.Typer()


def run(
    dataset: str = typer.Option(..., help="Dataset"),
    model: str = typer.Option(..., help="Model"),
    task: Task = typer.Option(..., help="Task type"),
    evaluation_file_path: str = typer.Option(..., help="Evaluation file path"),
    packages_file_path: str = typer.Option(..., help="Packages file path"),
    wandb_log_dir: str = typer.Option(..., help="Wandb log dir"),
    best_params_path: str = typer.Option(..., help="Best parameters path"),
) -> None:
    assert Path(wandb_log_dir).is_dir()
    submission_name = f"{dataset}_{model}"

    submission = Submission.from_local_disk(
        submission_name=submission_name,
        task=task,
        evaluation_file_path=evaluation_file_path,
        packages_file_path=packages_file_path,
        wandb_config_path=wandb_log_dir,
        best_params_path=best_params_path,
    )
    submission.save_json(SUBMISSIONS_PATH / submission_name)


typer.run(run)
