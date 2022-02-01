from typing import Dict

import datasets
import pandas as pd

from embeddings.transformation.transformation import Transformation


class ToPandasHuggingFaceCorpusTransformation(
    Transformation[datasets.DatasetDict, Dict[str, pd.DataFrame]]
):
    def transform(self, data: datasets.DatasetDict) -> Dict[str, pd.DataFrame]:
        out = {}
        for subset in data.keys():
            out[subset] = data[subset].to_pandas()
        return out
