from typing import Any, Dict, Optional, Union

import datasets

from embeddings.data.data_loader import HF_DATALOADERS, get_hf_dataloader
from embeddings.data.dataset import Dataset
from embeddings.pipeline import DOWNSAMPLE_SPLITS_TYPE, SAMPLE_MISSING_SPLITS_TYPE
from embeddings.pipeline.preprocessing_pipeline import PreprocessingPipeline
from embeddings.transformation.hf_transformation.downsample_transformation import (
    DownsampleHuggingFaceTransformation,
)
from embeddings.transformation.hf_transformation.drop_subset_transformation import (
    DropSubsetHuggingFaceCorpusTransformation,
)
from embeddings.transformation.hf_transformation.sampling_transformation import (
    SampleSplitsHuggingFaceTransformation,
)
from embeddings.transformation.transformation import DummyTransformation, Transformation
from embeddings.utils.hf_persister import HuggingFaceDatasetLocalPersister


class HuggingFacePreprocessingPipeline(
    PreprocessingPipeline[str, datasets.DatasetDict, datasets.DatasetDict]
):
    def __init__(
        self,
        dataset_name: str,
        persist_path: str,
        sample_missing_splits: Optional[SAMPLE_MISSING_SPLITS_TYPE] = None,
        downsample_splits: Optional[DOWNSAMPLE_SPLITS_TYPE] = None,
        ignore_test_subset: bool = False,
        seed: int = 441,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        dataset = Dataset(dataset_name, **load_dataset_kwargs if load_dataset_kwargs else {})
        data_loader: HF_DATALOADERS = get_hf_dataloader(dataset)

        transformation: Union[
            DummyTransformation[datasets.DatasetDict],
            Transformation[datasets.DatasetDict, datasets.DatasetDict],
        ] = DummyTransformation()

        if ignore_test_subset:
            transformation = transformation.then(
                DropSubsetHuggingFaceCorpusTransformation(subset="test")
            )

        if sample_missing_splits:
            transformation = transformation.then(
                SampleSplitsHuggingFaceTransformation(*sample_missing_splits, seed=seed)
            )

        if downsample_splits:
            transformation = transformation.then(
                DownsampleHuggingFaceTransformation(*downsample_splits, seed=seed)
            )

        transformation = transformation.persisting(
            HuggingFaceDatasetLocalPersister(path=persist_path)
        )
        super().__init__(dataset, data_loader, transformation)
