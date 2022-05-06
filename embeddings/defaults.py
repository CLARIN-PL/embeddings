import os
from pathlib import Path

EXPERIMENTAL_PATH = Path(os.path.dirname(__file__)).parent.absolute()

RESOURCES_PATH = EXPERIMENTAL_PATH / "resources"
DATASET_PATH = RESOURCES_PATH / "datasets"
RESULTS_PATH = RESOURCES_PATH / "results"
EVALUATIONS_PATH = RESOURCES_PATH / "evaluations"
SUBMISSIONS_PATH = RESOURCES_PATH / "submissions"
