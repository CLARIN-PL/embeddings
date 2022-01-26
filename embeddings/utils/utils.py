import copy
import importlib
import os.path
import pprint
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import yaml
from numpy import typing as nptyping

from embeddings.data.io import T_path

Numeric = Union[float, int]
PrimitiveTypes = Union[None, bool, int, float, str]
NDArrayInt = nptyping.NDArray[np.int_]


def import_from_string(dotted_path: str) -> Any:
    """
    The function taken from github.com/UKPLab/sentence-transformers.

    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError:
        msg = "%s doesn't look like a module path" % dotted_path
        raise ImportError(msg)

    try:
        module = importlib.import_module(dotted_path)
    except:
        module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)
        raise ImportError(msg)


def build_output_path(root: T_path, embedding_name: T_path, dataset_name: T_path) -> Path:
    """Builds output path using pattern {root}/{embedding_name}/{dataset_name}.
    Every "/" in the embedding/dataset name is replaced with  "__".
    E.g. "clarin-pl/nkjp-pos" -> "clarin-pl__nkjp-pos".

    Be aware that if passed paths are str (instead of Path) they are not checked if they exist
    and if they are dirs.
    """

    def _get_new_dir_name(embedding_or_dataset: T_path) -> str:
        if os.path.isdir(embedding_or_dataset):
            return Path(embedding_or_dataset).name
        else:
            assert isinstance(embedding_or_dataset, str)
            return embedding_or_dataset.replace("/", "__")

    for x in [embedding_name, dataset_name]:
        if isinstance(x, Path) and (x.is_file() or not x.exists()):
            raise ValueError(f"Path {x} is not correct.")

    embedding_name = _get_new_dir_name(embedding_name)
    dataset_name = _get_new_dir_name(dataset_name)
    return Path(root, embedding_name, dataset_name)


def format_eval_result(result: Dict[str, Any]) -> str:
    return pprint.pformat(result, sort_dicts=False)


def initialize_kwargs(
    default_kwargs: Dict[str, Any], user_kwargs: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    new_kwargs = copy.deepcopy(default_kwargs)
    new_kwargs.update(user_kwargs if user_kwargs else {})
    return new_kwargs


def read_yaml(filepath: Union[str, Path]) -> Any:
    with open(filepath, "r") as f:
        return yaml.safe_load(f)
