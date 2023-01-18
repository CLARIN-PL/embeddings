import datasets

from embeddings.transformation.transformation import Transformation


class ClassEncodeColumnTransformation(Transformation[datasets.DatasetDict, datasets.DatasetDict]):
    def __init__(
        self,
        column: str,
    ):
        self.column = column

    def transform(self, data: datasets.DatasetDict) -> datasets.DatasetDict:
        return data.class_encode_column(column=self.column)
