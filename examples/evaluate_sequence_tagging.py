import pprint
from pathlib import Path

import typer

from embeddings.defaults import RESULTS_PATH
from embeddings.evaluator.sequence_tagging_evaluator import SequenceTaggingEvaluator
from embeddings.pipeline.hugging_face_sequence_tagging import HuggingFaceSequenceTaggingPipeline

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

    evaluator = SequenceTaggingEvaluator()

    pipeline = HuggingFaceSequenceTaggingPipeline(
        embedding_name,
        dataset_name,
        input_column_name,
        target_column_name,
        output_path,
        hidden_size,
    )
    result = pipeline.run()
    typer.echo(pprint.pformat(result))


typer.run(run)
