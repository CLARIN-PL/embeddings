from pathlib import Path
from typing import Any

import pytest
from pandas.testing import assert_frame_equal  # type: ignore

from experimental.datasets.klej_hf_helper import (
    align_datasets,
    align_labels,
    load_huggingface_dataset,
    load_klej_dataset,
)

KLEJ_DATA_DIR = Path("../klej_data")


@pytest.mark.parametrize(
    "klej_dir_name, hf_name, label_col, load_hf_dataset_kwargs",
    [
        ("klej_nkjp-ner", "nkjp-ner", "target", {}),
        ("klej_cdsc-e", "cdsc", "entailment_judgment", {"name": "cdsc-e"}),
        ("klej_cdsc-r", "cdsc", "relatedness_score", {"name": "cdsc-r"}),
        ("klej_psc", "psc", "label", {}),
        ("klej_dyk", "dyk", "target", {}),
    ],
)
def test_klej_hugging_face_consistency(
    klej_dir_name: str, hf_name: str, label_col: str, load_hf_dataset_kwargs: Any
) -> None:
    hf_dataset = load_huggingface_dataset(hf_name, **load_hf_dataset_kwargs)
    klej_dataset = load_klej_dataset(klej_dir_name, klej_data_dir=KLEJ_DATA_DIR)
    for hf_data, klej_data in zip(hf_dataset, klej_dataset):
        if hf_data and klej_data is not None:
            klej_data = align_labels(klej_data, hf_data, label=label_col)
            assert set(hf_data.column_names) == set(klej_data.columns)
            hf_data, klej_data = align_datasets(hf_data, klej_data)
            assert_frame_equal(hf_data.reset_index(drop=True), klej_data.reset_index(drop=True))
