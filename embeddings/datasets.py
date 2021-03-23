import base64
import os
import zipfile
from typing import Optional, List, Union

import requests
from tqdm.auto import tqdm


class DatasetDownloader:
    def __init__(self, root_dir: str, url: str, filename: Optional[str] = None):
        self.root_dir = root_dir
        self.url = url
        self.filename = filename

        self._chunk_size = 1024

    @property
    def _dl_path(self) -> str:
        return os.path.join(self.root_dir, self.filename)

    def _download_file(self):
        r = requests.get(self.url, stream=True)
        assert r.status_code == 200

        if not self.filename:
            print(r.headers.get("Content-Disposition"))
            self.filename = r.headers.get(
                "Content-Disposition", "filename=ds"
            ).split("filename=")[1].replace('\"', '')

        filesize = int(r.headers.get('Content-Length', '0'))
        pbar = tqdm(total=filesize, unit='iB', unit_scale=True)
        os.makedirs(os.path.dirname(self._dl_path), exist_ok=True)

        with open(self._dl_path, 'wb') as f:
            for data in r.iter_content(chunk_size=self._chunk_size):
                f.write(data)
                pbar.update(len(data))

        pbar.close()

    def _unzip(self) -> List[str]:
        zf = zipfile.ZipFile(self._dl_path, 'r')
        zf.extractall(self.root_dir)
        zf.close()
        os.remove(self._dl_path)
        return [
            os.path.join(self.root_dir, it) for it in os.listdir(self.root_dir)
        ]

    def download(self) -> Union[str, List[str]]:
        self._download_file()
        if zipfile.is_zipfile(self._dl_path):
            return self._unzip()
        return self._dl_path


def create_onedrive_directdownload(onedrive_link: str) -> str:
    # Stąd ukradli: https://towardsdatascience.com/how-to-get-onedrive-direct-download-link-ecb52a62fee4
    data_b64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
    data_b64_str = (
        data_b64
        .decode('utf-8')
        .replace('/', '_')
        .replace('+', '-')
        .rstrip("=")
    )
    result = f"https://api.onedrive.com/v1.0/shares/u!" \
             f"{data_b64_str}/root/content"
    return result
