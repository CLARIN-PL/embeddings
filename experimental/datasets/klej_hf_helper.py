import csv
from pathlib import Path
from typing import Any, Optional

import datasets
import pandas as pd  # type: ignore
from datasets import Dataset, load_dataset


def load_huggingface_dataset(
    dataset_name_or_path: str, **load_dataset_kwargs: Any
) -> tuple[Dataset, Optional[Dataset], Dataset]:
    dataset = load_dataset(dataset_name_or_path, **load_dataset_kwargs)
    if len(dataset.keys()) == 2:
        train, val, test = dataset["train"], None, dataset["test"]
    else:
        train, val, test = dataset["train"], dataset["validation"], dataset["test"]
    return train, val, test


def load_klej_dataset(
    dataset_dir: str,
    klej_data_dir: Path,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    dataset_path = klej_data_dir.joinpath(dataset_dir)
    train = pd.read_csv(dataset_path.joinpath("train.tsv"), sep="\t", quoting=csv.QUOTE_NONE)
    try:
        val = pd.read_csv(dataset_path.joinpath("dev.tsv"), sep="\t", quoting=csv.QUOTE_NONE)
    except FileNotFoundError:
        val = None
    test = pd.read_csv(dataset_path.joinpath("test_features.tsv"), sep="\t", quoting=csv.QUOTE_NONE)
    return train, val, test


def align_labels(klej_data: pd.DataFrame, hf_data: datasets.Dataset, label: str) -> pd.DataFrame:
    if label in klej_data.columns:
        if pd.api.types.is_string_dtype(klej_data[label]):
            klej_data[label] = klej_data[label].apply(lambda x: hf_data.features[label].str2int(x))
    else:
        # test features in klej data do not have a label column (append one as in HF)
        klej_data[label] = -1
    return klej_data


def align_datasets(hf_data: Dataset, klej_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hf_data = hf_data.to_pandas()
    hf_data = hf_data[klej_data.columns]
    hf_data = hf_data.astype(klej_data.dtypes)
    return hf_data, klej_data
