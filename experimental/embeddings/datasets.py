import abc
import importlib
import os
import zipfile
from typing import Optional, List, Union

import requests
import srsly
from flair.datasets import ColumnCorpus
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from experimental.embeddings.converters import convert_jsonl_to_connl


def split_train_test_dev(data: List):
    train, test = train_test_split(data, test_size=0.4, random_state=1)
    dev, test = train_test_split(test, test_size=0.5, random_state=1)
    return {"train": train, "dev": dev, "test": test}


class Dataset:
    @abc.abstractmethod
    def to_flair_column_corpus(self) -> ColumnCorpus:
        pass


def get_dataset_cls(
    name: str, datasets_module: str = "experimental.embeddings.datasets"
):
    try:
        return getattr(importlib.import_module(datasets_module), name)
    except AttributeError:
        raise NotImplementedError(f"Dataset {name} not supported!")


class PromisesElectionsPLDataset(Dataset):
    def __init__(self, url: Optional[str] = None):
        super().__init__()
        self.ds_files = ["train.csv", "dev.csv", "test.csv"]
        self.url = url
        self.path = "resources/datasets/promises-elections-pl/"
        if not self._detect_files(path=self.path, files=self.ds_files):
            if url:
                print("Downloading and preprocessing data.")
                self._preprocess()
            else:
                raise ValueError("Can't download files without download url being set.")

    @staticmethod
    def _detect_files(path: str, files: List[str]) -> bool:
        return all([os.path.exists(os.path.join(path, file)) for file in files])

    def _download(self) -> str:
        downloader = DatasetDownloader(
            root_dir="resources/datasets/promises-elections-pl/",
            url=self.url,
        )
        return downloader.download()

    def _preprocess(self) -> None:
        path = self._download()
        data = list(srsly.read_jsonl(path))
        splitted_data = split_train_test_dev(data)
        for key, ds in splitted_data.items():
            convert_jsonl_to_connl(ds, out_path=os.path.join(self.path, f"{key}.csv"))
        os.remove(path)

    def to_flair_column_corpus(self) -> ColumnCorpus:
        return ColumnCorpus(
            data_folder=self.path,
            column_format={0: "text", 1: "tag"},
            train_file="train.csv",
            test_file="test.csv",
            dev_file="dev.csv",
        )


class DatasetDownloader:
    def __init__(self, root_dir: str, url: str, filename: Optional[str] = None):
        self.root_dir = root_dir
        self.url = url
        self.filename = filename

        self._chunk_size = 1024

    @property
    def _dl_path(self) -> str:
        return os.path.join(self.root_dir, self.filename)

    def _download_file(self) -> None:
        r = requests.get(self.url, stream=True)
        assert r.status_code == 200

        if not self.filename:
            print(r.headers.get("Content-Disposition"))
            self.filename = (
                r.headers.get("Content-Disposition", "filename=ds")
                .split("filename=")[1]
                .replace('"', "")
            )

        filesize = int(r.headers.get("Content-Length", "0"))
        pbar = tqdm(total=filesize, unit="iB", unit_scale=True)
        os.makedirs(os.path.dirname(self._dl_path), exist_ok=True)

        with open(self._dl_path, "wb") as f:
            for data in r.iter_content(chunk_size=self._chunk_size):
                f.write(data)
                pbar.update(len(data))

        pbar.close()

    def _unzip(self) -> List[str]:
        zf = zipfile.ZipFile(self._dl_path, "r")
        zf.extractall(self.root_dir)
        zf.close()
        os.remove(self._dl_path)
        return [os.path.join(self.root_dir, it) for it in os.listdir(self.root_dir)]

    def download(self) -> Union[str, List[str]]:
        self._download_file()
        if zipfile.is_zipfile(self._dl_path):
            return self._unzip()
        return self._dl_path
