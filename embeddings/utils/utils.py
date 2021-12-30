import copy
import importlib
from typing import Any, Dict, Optional, Union

import numpy as np
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

    Be aware that passed paths is str (instead of Path) are not checked if they exist and if
    they are dirs.
    """

    for x in [embedding_name, dataset_name]:
        if isinstance(x, Path) and (x.is_file() or not x.exists()):
            raise ValueError(f"Path {x} is not correct.")

    if os.path.isdir(embedding_name):
        embedding_name = Path(embedding_name).name
    else:
        assert isinstance(embedding_name, str)
        embedding_name = embedding_name.replace("/", "__")

    if os.path.isdir(dataset_name):
        dataset_name = Path(dataset_name).name
    else:
        assert isinstance(dataset_name, str)
        dataset_name = dataset_name.replace("/", "__")

    return Path(root, embedding_name, dataset_name)


def initialize_kwargs(
    default_kwargs: Dict[str, Any], user_kwargs: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    new_kwargs = copy.deepcopy(default_kwargs)
    new_kwargs.update(user_kwargs if user_kwargs else {})
    return new_kwargs
