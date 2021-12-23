import datasets

from embeddings.transformation.transformation import Transformation


class DropSubsetHuggingFaceCorpusTransformation(
    Transformation[datasets.DatasetDict, datasets.DatasetDict]
):
    HF_SUBSETS = ["validation", "test"]

    def __init__(self, subset: str) -> None:
        if subset not in self.HF_SUBSETS:
            raise ValueError(
                "Wrong subset name given in the argument - you can drop only validation or test subset."
            )
        self.subset = subset

    def transform(self, data: datasets.DatasetDict) -> datasets.DatasetDict:
        if self.subset in data:
            del data[self.subset]
        return data
