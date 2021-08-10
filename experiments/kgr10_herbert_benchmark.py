from pathlib import Path
from typing import Any, Dict, Optional

import datasets
import numpy as np
from flair.data import Corpus

from embeddings.data.hugging_face_data_loader import HuggingFaceDataLoader
from embeddings.data.hugging_face_dataset import HuggingFaceDataset
from embeddings.data.io import T_path
from embeddings.embedding.flair_embedding import FlairTransformerDocumentEmbedding
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.flair_task.text_classification import TextClassification
from embeddings.transformation.flair_transformation.classification_corpus_transformation import (
    ClassificationCorpusTransformation,
)
from embeddings.utils.json_dict_persister import JsonPersister

TASK_PARAMS = {"learning_rate": 0.01, "max_epochs": 20}


class HuggingFaceClassificationPipeline(
    StandardPipeline[str, datasets.DatasetDict, Corpus, Dict[str, np.ndarray], Dict[str, Any]]
):
    def __init__(
        self,
        embedding_name: str,
        dataset_name: str,
        input_column_name: str,
        target_column_name: str,
        output_path: T_path,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
    ):
        results_file_path = Path(".") / f"{embedding_name}-{dataset_name}.json"
        results_file_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = HuggingFaceDataset(dataset_name)
        data_loader = HuggingFaceDataLoader()
        transformation = ClassificationCorpusTransformation(input_column_name, target_column_name)
        embedding = FlairTransformerDocumentEmbedding(embedding_name)
        task = TextClassification(
            output_path, task_model_kwargs=task_model_kwargs, task_train_kwargs=task_train_kwargs
        )
        model = FlairModel(embedding, task)
        evaluator = TextClassificationEvaluator().persisting(
            JsonPersister(results_file_path.as_posix())
        )
        super().__init__(dataset, data_loader, transformation, model, evaluator)


pipeline_baseline = HuggingFaceClassificationPipeline(
    dataset_name="clarin-pl/polemo2-official",
    embedding_name="allegro/herbert-base-cased",
    input_column_name="text",
    target_column_name="target",
    output_path="herbert-base",
    task_model_kwargs={"loss_weights": {"plus": 2.0, "minus": 2.0}},
    task_train_kwargs=TASK_PARAMS,
)

pipeline_domain_fine_tuned = HuggingFaceClassificationPipeline(
    dataset_name="clarin-pl/polemo2-official",
    embedding_name="clarin-pl/herbert-kgr10",
    input_column_name="text",
    target_column_name="target",
    output_path="herbert-kgr10",
    task_model_kwargs={"loss_weights": {"plus": 2.0, "minus": 2.0}},
    task_train_kwargs=TASK_PARAMS,
)

print(pipeline_baseline.run())
print(pipeline_domain_fine_tuned.run())
