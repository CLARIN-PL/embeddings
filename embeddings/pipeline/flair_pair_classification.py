from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import datasets
from flair.data import Corpus

from embeddings.config.flair_config import (
    FlairTextClassificationBasicConfig,
    FlairTextClassificationConfig,
)
from embeddings.data.data_loader import HuggingFaceDataLoader
from embeddings.data.dataset import Dataset
from embeddings.data.io import T_path
from embeddings.embedding.flair_loader import FlairDocumentPoolEmbeddingLoader
from embeddings.evaluator.evaluation_results import Predictions, TextClassificationEvaluationResults
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.flair_task.text_pair_classification import TextPairClassification
from embeddings.transformation.flair_transformation.pair_classification_corpus_transformation import (
    PairClassificationCorpusTransformation,
)
from embeddings.transformation.flair_transformation.split_sample_corpus_transformation import (
    SampleSplitsFlairCorpusTransformation,
)
from embeddings.transformation.transformation import Transformation
from embeddings.utils.json_dict_persister import JsonPersister


class FlairPairClassificationPipeline(
    StandardPipeline[
        str,
        datasets.DatasetDict,
        Corpus,
        Predictions,
        TextClassificationEvaluationResults,
    ]
):
    def __init__(
        self,
        embedding_name: T_path,
        dataset_name: str,
        input_columns_names_pair: Tuple[str, str],
        target_column_name: str,
        output_path: T_path,
        model_type_reference: str = "",
        evaluation_filename: str = "evaluation.json",
        config: FlairTextClassificationConfig = FlairTextClassificationBasicConfig(),
        sample_missing_splits: Optional[Tuple[Optional[float], Optional[float]]] = None,
        seed: int = 441,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        output_path = Path(output_path)
        dataset = Dataset(dataset_name, **load_dataset_kwargs if load_dataset_kwargs else {})
        data_loader = HuggingFaceDataLoader()
        transformation: Union[
            Transformation[datasets.DatasetDict, Corpus], Transformation[Corpus, Corpus]
        ]
        transformation = PairClassificationCorpusTransformation(
            input_columns_names_pair, target_column_name
        )
        if sample_missing_splits:
            transformation = transformation.then(
                SampleSplitsFlairCorpusTransformation(*sample_missing_splits, seed=seed)
            )

        embedding_loader = FlairDocumentPoolEmbeddingLoader(embedding_name, model_type_reference)
        embedding = embedding_loader.get_embedding(
            config.document_embedding_cls, **config.load_model_kwargs
        )

        task = TextPairClassification(
            output_path,
            task_model_kwargs=config.task_model_kwargs,
            task_train_kwargs=config.task_train_kwargs,
        )
        model = FlairModel(embedding, task)
        evaluator = TextClassificationEvaluator().persisting(
            JsonPersister(path=output_path.joinpath(evaluation_filename))
        )
        super().__init__(dataset, data_loader, transformation, model, evaluator)
