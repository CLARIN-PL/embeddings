import os
from pathlib import Path

EXPERIMENTAL_PATH = Path(os.path.dirname(__file__)).parent.absolute()

RESOURCES_PATH = EXPERIMENTAL_PATH.joinpath("resources")
DATASET_PATH = RESOURCES_PATH.joinpath("datasets")
RESULTS_PATH = RESOURCES_PATH.joinpath("results")
