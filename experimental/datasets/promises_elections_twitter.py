from pathlib import Path
from typing import Optional

import srsly
from flair.datasets import ColumnCorpus
from spacy.lang import pl

from embeddings.data.io import T_path
from experimental.datasets.base import BaseDataset
from experimental.datasets.utils.converters import convert_spacy_jsonl_to_connl_bilou
from experimental.datasets.utils.misc import (
    DatasetDownloader,
    all_files_exists,
    split_train_test_dev,
)


class PromisesElectionsPLDataset(BaseDataset):
    def __init__(
        self,
        url: Optional[str] = None,
        output_path: T_path = "resources/datasets/promises-elections-pl/",
    ):
        super().__init__()
        self.ds_files = ["train.csv", "dev.csv", "test.csv"]
        self.url = url
        self.path = Path(output_path)
        if not all_files_exists(path=self.path, files=self.ds_files):
            if url:
                print("Downloading and preprocessing data.")
                self._preprocess()
            else:
                raise ValueError("Can't download files without download url being set.")

    def _download(self) -> Path:
        assert isinstance(self.url, str)
        downloader = DatasetDownloader(
            root_dir=str(self.path.joinpath("src")),
            url=self.url,
        )
        downloaded = downloader.download()
        if len(downloaded) != 1:
            raise ValueError(f"Download failed. Expected one file, got {len(downloaded)}")

        return downloaded[0]

    def _preprocess(self) -> None:
        path = self._download()
        data = list(srsly.read_jsonl(path))
        splitted_data = split_train_test_dev(data)
        nlp = pl.Language()
        for key, ds in splitted_data.items():
            convert_spacy_jsonl_to_connl_bilou(
                ds, nlp=nlp, out_path=str(self.path.joinpath(f"{key}.csv"))
            )

    def to_flair_column_corpus(self) -> ColumnCorpus:
        return ColumnCorpus(
            data_folder=self.path,
            column_format={0: "text", 1: "tag"},
            train_file="train.csv",
            test_file="test.csv",
            dev_file="dev.csv",
        )
