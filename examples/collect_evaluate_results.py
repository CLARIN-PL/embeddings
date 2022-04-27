from pprint import pprint

import wandb

from embeddings.defaults import EVALUATIONS_PATH
from embeddings.evaluator.submission import Submission

ENTITY = "albert__"
PROJECT = "test-hps-run-13"
ROOT = EVALUATIONS_PATH

TASK = "sequence_labeling"

api = wandb.Api()
# Call to untyped function "runs" in typed context
runs = api.runs(
    ENTITY + "/" + PROJECT, filters={"config.predict_subset": "LightingDataModuleSubset.TEST"}
)  # type: ignore

for run in runs:
    submission = Submission.from_run("sub name", run, TASK)
    pprint(submission)
    submission.save_json()
    pass
