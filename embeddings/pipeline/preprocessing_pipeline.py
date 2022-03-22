from typing import Generic

from embeddings.data.data_loader import DataLoader
from embeddings.data.dataset import BaseDataset, Data
from embeddings.pipeline.pipeline import Pipeline
from embeddings.pipeline.standard_pipeline import LoaderResult, TransformationResult
from embeddings.transformation.transformation import Transformation


class PreprocessingPipeline(
    Pipeline[TransformationResult], Generic[Data, LoaderResult, TransformationResult]
):
    def __init__(
        self,
        dataset: BaseDataset[Data],
        data_loader: DataLoader[Data, LoaderResult],
        transformation: Transformation[LoaderResult, TransformationResult],
    ) -> None:
        self.dataset = dataset
        self.data_loader = data_loader
        self.transformation = transformation

    def run(self) -> TransformationResult:
        loaded_data = self.data_loader.load(self.dataset)
        result = self.transformation.transform(loaded_data)
        return result
