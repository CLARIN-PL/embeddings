from pathlib import Path
from typing import Optional, List

import srsly
from flair.datasets import ColumnCorpus

from experimental.datasets.base import Dataset
from experimental.datasets.utils.converters import convert_spacy_jsonl_to_connl
from experimental.datasets.utils.misc import split_train_test_dev, DatasetDownloader


class PromisesElectionsPLDataset(Dataset):
    def __init__(
        self,
        url: Optional[str] = None,
        output_path: str = "resources/datasets/promises-elections-pl/",
    ):
        super().__init__()
        self.ds_files = ["train.csv", "dev.csv", "test.csv"]
        self.url = url
        self.path = Path(output_path)
        if not self._detect_files(path=self.path, files=self.ds_files):
            if url:
                print("Downloading and preprocessing data.")
                self._preprocess()
            else:
                raise ValueError("Can't download files without download url being set.")

    @staticmethod
    def _detect_files(path: Path, files: List[str]) -> bool:
        return all([path.joinpath(file).exists() for file in files])

    def _download(self) -> Path:
        assert isinstance(self.url, str)
        downloader = DatasetDownloader(
            root_dir=str(self.path.joinpath("src")),
            url=self.url,
        )
        downloaded = downloader.download()
        if len(downloaded) != 1:
            raise ValueError(
                f"Download failed. Expected one file, got {len(downloaded)}"
            )

        return downloaded[0]

    def _preprocess(self) -> None:
        path = self._download()
        data = list(srsly.read_jsonl(path))
        splitted_data = split_train_test_dev(data)
        for key, ds in splitted_data.items():
            convert_spacy_jsonl_to_connl(
                ds, out_path=str(self.path.joinpath(f"{key}.csv"))
            )

    def to_flair_column_corpus(self) -> ColumnCorpus:
        return ColumnCorpus(
            data_folder=self.path,
            column_format={0: "text", 1: "tag"},
            train_file="train.csv",
            test_file="test.csv",
            dev_file="dev.csv",
        )
