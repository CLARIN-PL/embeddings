from typing import Any, Dict, Optional, Tuple, Union

import datasets

from embeddings.data.data_loader import HuggingFaceDataLoader
from embeddings.data.dataset import HuggingFaceDataset
from embeddings.pipeline.preprocessing_pipeline import PreprocessingPipeline
from embeddings.transformation.hf_transformation.drop_subset_transformation import (
    DropSubsetHuggingFaceCorpusTransformation,
)
from embeddings.transformation.transformation import EmptyTransformation, Transformation
from embeddings.utils.hf_persister import HuggingFaceDatasetLocalPersister


class HuggingFaceTextClassificationPreprocessingPipeline(
    PreprocessingPipeline[str, datasets.DatasetDict, datasets.DatasetDict]
):
    def __init__(
        self,
        dataset_name: str,
        persist_path: str,
        sample_missing_splits: Optional[Tuple[Optional[float], Optional[float]]] = None,
        ignore_test_subset: bool = False,
        seed: int = 441,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        dataset = HuggingFaceDataset(
            dataset_name, **load_dataset_kwargs if load_dataset_kwargs else {}
        )
        data_loader = HuggingFaceDataLoader()

        transformation: Union[
            EmptyTransformation[datasets.DatasetDict],
            Transformation[datasets.DatasetDict, datasets.DatasetDict],
        ] = EmptyTransformation()

        # # if ignore_test_subset:
        #     transformation = transformation.then(
        #         SampleSplitsFlairCorpusTransformation(*sample_missing_splits, seed=seed)
        #     )

        if ignore_test_subset:
            transformation = transformation.then(
                DropSubsetHuggingFaceCorpusTransformation(subset="test")
            )
        transformation = transformation.persisting(
            HuggingFaceDatasetLocalPersister(path=persist_path)
        )
        super().__init__(dataset, data_loader, transformation)
