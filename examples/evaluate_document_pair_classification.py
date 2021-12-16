import pprint
from pathlib import Path
from typing import Tuple

import typer

from embeddings.defaults import RESULTS_PATH
from embeddings.pipeline.flair_pair_classification import FlairPairClassificationPipeline

app = typer.Typer()


def run(
    embedding_name: str = typer.Option(
        "allegro/herbert-base-cased", help="Hugging Face embedding model name or path."
    ),
    dataset_name: str = typer.Option(
        "clarin-pl/cst-wikinews", help="Hugging Face dataset name or path."
    ),
    input_columns_names_pair: Tuple[str, str] = typer.Option(
        ("sentence_1", "sentence_2"), help="Pair of column names that contain texts to classify."
    ),
    target_column_name: str = typer.Option(
        "label", help="Column name that contains label for classification."
    ),
    root: str = typer.Option(RESULTS_PATH.joinpath("document_pair_classification")),
) -> None:
    typer.echo(pprint.pformat(locals()))

    output_path = Path(root, embedding_name, dataset_name)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline = FlairPairClassificationPipeline(
        embedding_name, dataset_name, input_columns_names_pair, target_column_name, output_path
    )
    result = pipeline.run()
    typer.echo(pprint.pformat(result))


typer.run(run)
