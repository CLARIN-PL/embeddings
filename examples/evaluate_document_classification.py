from pathlib import Path
import pprint

import typer

from embeddings.data.hugging_face_data_loader import HuggingFaceDataLoader
from embeddings.data.hugging_face_dataset import HuggingFaceDataset
from embeddings.embedding.flair_embedding import FlairTransformerDocumentEmbedding
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.flair.text_classification import TextClassification
from embeddings.transformation.flair_transformation.classification_corpus_transformation import (
    ClassificationCorpusTransformation,
)
from experimental.defaults import RESULTS_PATH

app = typer.Typer()


def run(
    embedding_name: str = typer.Option(
        "allegro/herbert-base-cased", help="Hugging Face embedding model name or path."
    ),
    dataset_name: str = typer.Option(
        "clarin-pl/polemo2-official", help="Hugging Face dataset name or path."
    ),
    input_column_name: str = typer.Option(
        "text", help="Column name that contains text to classify."
    ),
    target_column_name: str = typer.Option(
        "target", help="Column name that contains label for classification."
    ),
    root: str = typer.Option(RESULTS_PATH.joinpath("document_classification")),
) -> None:
    typer.echo(pprint.pformat(locals()))

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
    typer.echo(pprint.pformat(result))


typer.run(run)
