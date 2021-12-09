import pprint
from pathlib import Path

import typer

from embeddings.defaults import RESULTS_PATH
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline


def run(
    embedding_name: str = typer.Option(
        "allegro/herbert-base-cased", help="Hugging Face embedding model name or path."
    ),
    dataset_name: str = typer.Option(
        "clarin-pl/polemo2-official", help="Hugging Face dataset name or path."
    ),
    input_columns_name: str = typer.Option(
        "text", help="Pair of column names that contain texts to classify."
    ),
    target_column_name: str = typer.Option(
        "target", help="Column name that contains label for classification."
    ),
    root: str = typer.Option(RESULTS_PATH.joinpath("lightning_sequence_classification")),
) -> None:
    typer.echo(pprint.pformat(locals()))

    output_path = Path(root, embedding_name, dataset_name)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline = LightningClassificationPipeline(
        embedding_name=embedding_name,
        dataset_name=dataset_name,
        input_column_name=input_columns_name,
        target_column_name=target_column_name,
        output_path=root,
    )

    result = pipeline.run()
    typer.echo(pprint.pformat(result))


typer.run(run)
