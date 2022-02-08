from typing import Dict

import pandas as pd

from embeddings.transformation.transformation import Transformation


class RenameInputColumnsTransformation(
    Transformation[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
):
    def __init__(self, input_column_name: str, target_column_name: str):
        self.input_column_name = input_column_name
        self.target_column_name = target_column_name

    def transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        for subset in data:
            data[subset].rename(
                {self.input_column_name: "x", self.target_column_name: "y"},
                axis="columns",
                inplace=True,
            )
        return data
