import pprint
from pathlib import Path

import typer

from embeddings.defaults import RESULTS_PATH
from embeddings.pipeline.flair_classification import FlairClassificationPipeline

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

    pipeline = FlairClassificationPipeline(
        embedding_name, dataset_name, input_column_name, target_column_name, output_path
    )
    result = pipeline.run()
    typer.echo(pprint.pformat(result))


typer.run(run)
