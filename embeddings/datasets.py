import os
from zipfile import ZipFile

import requests
from tqdm.auto import tqdm


class KLEJDatasetDownloader:
    def __init__(self, root_dir: str, url: str):
        self.root_dir = root_dir
        self.url = url

        self._zip_filename = 'dataset.zip'
        self._zip_out_path = os.path.join(self.root_dir, self._zip_filename)
        self._chunk_size = 1024

    def _download_zip(self):
        r = requests.get(self.url, stream=True)
        assert r.status_code == 200

        filesize = int(r.headers.get('Content-Length', '0'))
        pbar = tqdm(total=filesize, unit='iB', unit_scale=True)
        os.makedirs(os.path.dirname(self._zip_out_path), exist_ok=True)

        with open(self._zip_out_path, 'wb') as f:
            for data in r.iter_content(chunk_size=self._chunk_size):
                f.write(data)
                pbar.update(len(data))

        pbar.close()

    def _unzip(self):
        zf = ZipFile(self._zip_out_path, 'r')
        zf.extractall(self.root_dir)
        zf.close()
        os.remove(self._zip_out_path)

    def download(self):
        self._download_zip()
        self._unzip()
