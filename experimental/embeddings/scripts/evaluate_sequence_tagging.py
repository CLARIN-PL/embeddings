from pathlib import Path
from pprint import pprint

import typer

from embeddings.data.hugging_face_data_loader import HuggingFaceDataLoader
from embeddings.data.hugging_face_dataset import HuggingFaceDataset
from embeddings.embedding.flair_embedding import FlairTransformerWordEmbedding
from embeddings.evaluator.sequence_tagging_evaluator import SequenceTaggingEvaluator
from embeddings.model.flair_model import FlairModel
from embeddings.pipeline.standard_pipeline import StandardPipeline
from embeddings.task.flair.sequence_tagging import SequenceTagging
from embeddings.transformation.flair.column_corpus_transformation import ColumnCorpusTransformation
from experimental.defaults import RESULTS_PATH

app = typer.Typer()


def run(
    embedding_name: str = typer.Option(..., help="Hugging Face embedding model name or path."),
    dataset_name: str = typer.Option(..., help="Hugging Face dataset name."),
    input_column_name: str = typer.Option(...),
    target_column_name: str = typer.Option(...),
    root: str = typer.Option(RESULTS_PATH.joinpath("document_classification")),
    hidden_size: int = typer.Option(..., help="Number of hidden states in RNN."),
) -> None:
    pprint(locals())

    output_path = Path(root, embedding_name, dataset_name)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = HuggingFaceDataset(dataset_name)
    data_loader = HuggingFaceDataLoader()
    transformation = ColumnCorpusTransformation(input_column_name, target_column_name)
    embedding = FlairTransformerWordEmbedding(embedding_name)
    task = SequenceTagging(output_path, hidden_size=256)
    model = FlairModel(embedding, task)
    evaluator = SequenceTaggingEvaluator()

    pipeline = StandardPipeline(dataset, data_loader, transformation, model, evaluator)
    result = pipeline.run()
    typer.echo(result)


typer.run(run)
