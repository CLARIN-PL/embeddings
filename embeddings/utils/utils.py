import copy
import importlib
import os.path
import pprint
import zipfile
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pkg_resources
import requests
import yaml
from numpy import typing as nptyping
from tqdm.auto import tqdm

from embeddings.data.io import T_path
from embeddings.evaluator.evaluation_results import EvaluationResults
from embeddings.utils.loggers import get_logger

Numeric = Union[float, int]
PrimitiveTypes = Union[None, bool, int, float, str]
NDArrayInt = nptyping.NDArray[np.int_]


_logger = get_logger(__name__)


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


def build_output_path(
    root: T_path,
    embedding_name: T_path,
    dataset_name: T_path,
    timestamp_subdir: bool = True,
    mkdirs: bool = True,
) -> Path:
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
            return standardize_name(embedding_or_dataset)

    for x in [embedding_name, dataset_name]:
        if isinstance(x, Path) and (x.is_file() or not x.exists()):
            raise ValueError(f"Path {x} is not correct.")

    embedding_name = _get_new_dir_name(embedding_name)
    dataset_name = _get_new_dir_name(dataset_name)
    path = Path(root, embedding_name, dataset_name)
    if timestamp_subdir:
        path = path.joinpath(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if mkdirs:
        path.mkdir(exist_ok=True, parents=True)
    return path


def standardize_name(text: str) -> str:
    if "/" in text:
        cleaned_text = text.replace("/", "__")
        _logger.warning(
            f"String '{text}' contains '/'. Replacing it with '__'. Cleaned_text: {cleaned_text}."
        )
        return cleaned_text
    else:
        return text


def format_eval_results(result: EvaluationResults) -> str:
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


def download_file(url: str, chunk_size: int = 1024) -> Tuple[Any, str]:
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise RuntimeError(
            f"Error while downloading file, response code status: {r.status_code}. "
            f"Check download url."
        )

    filename = (
        r.headers.get("Content-Disposition", "filename=ds").split("filename=")[1].replace('"', "")
    )
    filesize = int(r.headers.get("Content-Length", "0"))

    pbar = tqdm(total=filesize, unit="iB", unit_scale=True)
    tmp_file = NamedTemporaryFile(delete=False)

    for data in r.iter_content(chunk_size=chunk_size):
        tmp_file.write(data)
        pbar.update(len(data))

    pbar.close()
    tmp_file.seek(0)
    return tmp_file, filename


def get_installed_packages() -> List[str]:
    return sorted([f"{p.key}=={p.version}" for p in pkg_resources.working_set])


def compress_and_remove(filepath: T_path) -> None:
    filepath = Path(filepath)
    with zipfile.ZipFile(
        filepath.with_name(filepath.name + ".zip"), mode="w", compression=zipfile.ZIP_DEFLATED
    ) as arc:
        arc.write(filepath, arcname=filepath.name)
    filepath.unlink()
