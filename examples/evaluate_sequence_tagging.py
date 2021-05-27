from pathlib import Path
import pprint

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
    embedding_name: str = typer.Option(
        "allegro/herbert-base-cased", help="Hugging Face embedding model name or path."
    ),
    dataset_name: str = typer.Option(
        "clarin-pl/nkjp-pos", help="Hugging Face dataset name or path."
    ),
    input_column_name: str = typer.Option(
        "tokens", help="Column name that contains text to classify."
    ),
    target_column_name: str = typer.Option(
        "pos_tags", help="Column name that contains tag labels for sequence tagging."
    ),
    root: str = typer.Option(RESULTS_PATH.joinpath("sequence_tagging")),
    hidden_size: int = typer.Option(256, help="Number of hidden states in RNN."),
) -> None:
    typer.echo(pprint.pformat(locals()))

    output_path = Path(root, embedding_name, dataset_name)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = HuggingFaceDataset(dataset_name)
    data_loader = HuggingFaceDataLoader()
    transformation = ColumnCorpusTransformation(input_column_name, target_column_name)
    embedding = FlairTransformerWordEmbedding(embedding_name)
    task = SequenceTagging(output_path, hidden_size=hidden_size)
    model = FlairModel(embedding, task)
    evaluator = SequenceTaggingEvaluator()

    pipeline = StandardPipeline(dataset, data_loader, transformation, model, evaluator)
    result = pipeline.run()
    typer.echo(pprint.pformat(result))


typer.run(run)
