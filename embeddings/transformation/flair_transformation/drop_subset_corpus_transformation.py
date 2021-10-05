from flair.data import Corpus

from embeddings.transformation.transformation import Transformation


class DropSubsetFlairCorpusTransformation(Transformation[Corpus, Corpus]):
    FLAIR_SUBSETS = ["dev", "test"]

    def __init__(self, subset: str):
        assert subset in self.FLAIR_SUBSETS
        self.subset = subset

    def transform(self, data: Corpus) -> Corpus:
        corpus_kwargs = {"train": data.train, self.subset: None, "sample_missing_splits": False}
        if self.subset == "dev":
            corpus_kwargs["test"] = data.test
        else:  # self.subset == "test"
            corpus_kwargs["dev"] = data.dev
        return Corpus(**corpus_kwargs)