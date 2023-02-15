import pprint
from itertools import groupby
from pathlib import Path

import typer
import wandb

from embeddings.defaults import SUBMISSIONS_PATH
from embeddings.evaluator.evaluation_results import Task
from embeddings.evaluator.submission import Submission
from embeddings.evaluator.submission_utils import filter_hps_summary_runs, filter_retrains

app = typer.Typer()


def main(
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
    runs = api.runs(entity + "/" + project)

    embeddings = {run.config["embedding_name_or_path"] for run in runs}
    hps_summary_runs = {
        run.config["embedding_name_or_path"]: run for run in filter_hps_summary_runs(runs)
    }
    for key, run_group in groupby(
        filter_retrains(runs), lambda x: x.config["embedding_name_or_path"]  # type: ignore
    ):
        [run] = run_group
        submission = Submission.from_wandb(key, run, hps_summary_runs[key], task)
        pprint.pprint(submission)
        submission.save_json(root)


typer.run(main)
