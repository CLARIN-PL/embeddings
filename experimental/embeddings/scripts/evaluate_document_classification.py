from pathlib import Path
from pprint import pprint

import typer

from embeddings.data.hugging_face_data_loader import HuggingFaceDataLoader
from embeddings.data.hugging_face_dataset import HuggingFaceDataset
from embeddings.embedding.flair_embedding import FlairTransformerDocumentEmbedding
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.flair.text_classification import TextClassification
from embeddings.transformation.flair.classification_corpus_transformation import (
    ClassificationCorpusTransformation,
)
from experimental.defaults import RESULTS_PATH

app = typer.Typer()


def run(
    embedding_name: str = typer.Option(..., help="Hugging Face embedding model name or path."),
    dataset_name: str = typer.Option(..., help="Hugging Face dataset name."),
    input_column_name: str = typer.Option("sentence"),
    target_column_name: str = typer.Option("target"),
    root: str = typer.Option(RESULTS_PATH.joinpath("document_classification")),
) -> None:
    typer.echo(locals())

    output_path = Path(root, embedding_name, dataset_name)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = HuggingFaceDataset(dataset_name)
    data_loader = HuggingFaceDataLoader()
    transformation = ClassificationCorpusTransformation(input_column_name, target_column_name)
    embedding = FlairTransformerDocumentEmbedding(embedding_name)
    task = TextClassification(output_path)
    model = FlairModel(embedding, task)
    evaluator = TextClassificationEvaluator()

    pipeline = StandardPipeline(dataset, data_loader, transformation, model, evaluator)
    result = pipeline.run()
    typer.echo(result)


typer.run(run)
