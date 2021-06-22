import importlib
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Tuple

import requests
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from embeddings.data.io import T_path


def split_train_test_dev(data: List[Any]) -> Dict[str, List[Any]]:
    train, test = train_test_split(data, test_size=0.4, random_state=1)
    dev, test = train_test_split(test, test_size=0.5, random_state=1)
    return {"train": train, "dev": dev, "test": test}


def get_dataset_cls(name: str, datasets_module: str = "experimental.datasets") -> Any:
    try:
        return getattr(importlib.import_module(datasets_module), name)
    except AttributeError:
        raise NotImplementedError(f"Dataset {name} not supported!")


def download_file(url: str, chunk_size: int = 1024) -> Tuple[Any, str]:
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise RuntimeError(
            f"Error while downloading file, response code status: {r.status_code}. Check download url."
        )

    filename = (
        r.headers.get("Content-Disposition", "filename=ds").split("filename=")[1].replace('"', "")
    )
    filesize = int(r.headers.get("Content-Length", "0"))

    pbar = tqdm(total=filesize, unit="iB", unit_scale=True)
    tmp_file = NamedTemporaryFile()

    for data in r.iter_content(chunk_size=chunk_size):
        tmp_file.write(data)
        pbar.update(len(data))

    pbar.close()
    tmp_file.seek(0)
    return tmp_file, filename


def unzip_file(path: str, out_dir: str) -> List[Path]:
    output_dir = Path(out_dir)
    if output_dir.exists():
        raise FileExistsError(f"Given output path {out_dir} exists.")
    zf = zipfile.ZipFile(path, "r")
    zf.extractall(str(output_dir))
    zf.close()
    return list(output_dir.iterdir())


def all_files_exists(path: T_path, files: List[str]) -> bool:
    path = Path(path)
    return all([path.joinpath(file).exists() for file in files])


class DatasetDownloader:
    def __init__(self, root_dir: str, url: str, filename: Optional[str] = None):
        self.root_dir = root_dir
        self.url = url
        self.filename = filename

    def download(self) -> List[Path]:
        tmp_dl_file, filename = download_file(self.url)
        if not self.filename:
            self.filename = filename

        output_paths = []
        if zipfile.is_zipfile(tmp_dl_file.name):
            output_paths.extend(unzip_file(path=tmp_dl_file.name, out_dir=self.root_dir))
        else:
            root_dir = Path(self.root_dir)
            root_dir.mkdir(parents=True)
            output_paths.append(root_dir.joinpath(self.filename))
            with output_paths[-1].open(mode="wb") as f:
                f.write(tmp_dl_file.read())

        tmp_dl_file.close()
        return output_paths
