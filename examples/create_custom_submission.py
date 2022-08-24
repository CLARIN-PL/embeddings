from typing import Any

import datasets
import numpy as np
from datasets import DatasetDict
from numpy import typing as nptyping

from embeddings.evaluator.evaluation_results import Predictions
from embeddings.evaluator.leaderboard import get_dataset_task
from embeddings.evaluator.submission import AveragedSubmission
from embeddings.utils.utils import get_installed_packages

DATASET_NAME = "clarin-pl/polemo2-official"
TARGET_COLUMN_NAME = "target"

hparams = {"hparam_name_1": 0.2, "hparam_name_2": 0.1}  # put your hyperparameters here!

dataset = datasets.load_dataset(DATASET_NAME)
assert isinstance(dataset, DatasetDict)
y_true: nptyping.NDArray[Any] = np.array(dataset["test"][TARGET_COLUMN_NAME])
# put your predictions from multiple runs below!
predictions = [
    Predictions(y_true=y_true, y_pred=np.random.randint(low=0, high=4, size=len(y_true)))
    for _ in range(5)
]

# make sure you are running on a training env or put exported packages below!
packages = get_installed_packages()
submission = AveragedSubmission.from_predictions(
    submission_name="your_submission_name",  # put your submission here!
    dataset_name=DATASET_NAME,
    dataset_version=dataset["train"].info.version.version_str,
    embedding_name="your_embedding_model",  # put your embedding name here!
    predictions=predictions,
    hparams=hparams,
    packages=packages,
    task=get_dataset_task(DATASET_NAME),
)

submission.save_json()
