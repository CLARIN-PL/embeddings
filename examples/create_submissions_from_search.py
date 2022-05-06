import pprint
from enum import Enum
from pathlib import Path

import typer
import wandb

from embeddings.defaults import SUBMISSIONS_PATH
from embeddings.evaluator.submission import Submission


class Task(str, Enum):
    sequence_labeling = "sequence_labeling"
    text_classification = "text_classification"


app = typer.Typer()


def run(
    entity: str = typer.Option(..., help="Wandb entity."),
    project: str = typer.Option(..., help="Wandb project."),
    task: Task = typer.Option(...),
    root: Path = typer.Option(
        SUBMISSIONS_PATH,
        help="Dir path to save a submission.",
        file_okay=False,
        dir_okay=True,
        writable=True,
    ),
) -> None:
    typer.echo(pprint.pformat(locals()))
    api = wandb.Api()
    # Call to untyped function "runs" in typed context
    runs = api.runs(
        entity + "/" + project, filters={"config.predict_subset": "LightingDataModuleSubset.TEST"}
    )  # type: ignore
    for run in runs:
        submission_name = run.name
        submission = Submission.from_wandb_run(run.name, run, task)
        pprint.pprint(submission)
        submission.save_json(root)


typer.run(run)
