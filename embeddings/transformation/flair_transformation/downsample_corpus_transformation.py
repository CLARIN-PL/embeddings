from flair.data import Corpus

from embeddings.transformation.transformation import Transformation


class DownsampleFlairCorpusTransformation(Transformation[Corpus, Corpus]):
    def __init__(
        self,
        percentage: float,
        downsample_train: bool = True,
        downsample_dev: bool = True,
        downsample_test: bool = True,
    ):
        self.percentage = percentage
        self.downsample_train = downsample_train
        self.downsample_dev = downsample_dev
        self.downsample_test = downsample_test

    def transform(self, data: Corpus) -> Corpus:
        return data.downsample(
            percentage=self.percentage,
            downsample_train=self.downsample_train,
            downsample_dev=self.downsample_dev,
            downsample_test=self.downsample_test,
        )
