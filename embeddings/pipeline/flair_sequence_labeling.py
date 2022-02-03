from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import datasets
from flair.data import Corpus
from numpy import typing as nptyping

from embeddings.data.data_loader import HuggingFaceDataLoader
from embeddings.data.dataset import HuggingFaceDataset
from embeddings.data.io import T_path
from embeddings.embedding.auto_flair import AutoFlairWordEmbedding
from embeddings.evaluator.sequence_labeling_evaluator import SequenceLabelingEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.flair_task.sequence_labeling import SequenceLabeling
from embeddings.transformation.flair_transformation.column_corpus_transformation import (
    ColumnCorpusTransformation,
)
from embeddings.transformation.flair_transformation.split_sample_corpus_transformation import (
    SampleSplitsFlairCorpusTransformation,
)
from embeddings.transformation.transformation import Transformation
from embeddings.utils.json_dict_persister import JsonPersister


class FlairSequenceLabelingPipeline(
    StandardPipeline[
        str, datasets.DatasetDict, Corpus, Dict[str, nptyping.NDArray[Any]], Dict[str, Any]
    ]
):
    def __init__(
        self,
        embedding_name: str,
        dataset_name: str,
        input_column_name: str,
        target_column_name: str,
        output_path: T_path,
        hidden_size: int,
        evaluation_filename: str = "evaluation.json",
        evaluation_mode: SequenceLabelingEvaluator.EvaluationMode = SequenceLabelingEvaluator.EvaluationMode.CONLL,
        tagging_scheme: Optional[SequenceLabelingEvaluator.TaggingScheme] = None,
        sample_missing_splits: Optional[Tuple[Optional[float], Optional[float]]] = None,
        seed: int = 441,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        output_path = Path(output_path)
        dataset = HuggingFaceDataset(
            dataset_name, **load_dataset_kwargs if load_dataset_kwargs else {}
        )
        data_loader = HuggingFaceDataLoader()
        transformation: Union[
            Transformation[datasets.DatasetDict, Corpus], Transformation[Corpus, Corpus]
        ]
        transformation = ColumnCorpusTransformation(input_column_name, target_column_name)
        if sample_missing_splits:
            transformation = transformation.then(
                SampleSplitsFlairCorpusTransformation(*sample_missing_splits, seed=seed)
            )

        embedding = AutoFlairWordEmbedding.from_hub(embedding_name)
        task = SequenceLabeling(
            output_path,
            hidden_size=hidden_size,
            task_model_kwargs=task_model_kwargs,
            task_train_kwargs=task_train_kwargs,
        )
        model = FlairModel(embedding, task)
        evaluator = SequenceLabelingEvaluator(
            evaluation_mode=evaluation_mode, tagging_scheme=tagging_scheme
        ).persisting(JsonPersister(path=output_path.joinpath(evaluation_filename)))
        super().__init__(dataset, data_loader, transformation, model, evaluator)
