import pprint
from pathlib import Path

import typer

from embeddings.defaults import DATASET_PATH
from embeddings.pipeline.preprocessing_pipeline import FlairSequenceLabelingPreprocessingPipeline

app = typer.Typer()


def run(
    embedding_name: str = typer.Option(
        "allegro/herbert-base-cased", help="Hugging Face embedding model name or path."
    ),
    dataset_name: str = typer.Option(
        "clarin-pl/kpwr-ner", help="Hugging Face dataset name or path."
    ),
    input_column_name: str = typer.Option(
        "tokens", help="Column name that contains text to classify."
    ),
    target_column_name: str = typer.Option(
        "ner", help="Column name that contains tag labels for POS tagging."
    ),
    root: str = typer.Option(DATASET_PATH.joinpath("pos_tagging")),
) -> None:
    typer.echo(pprint.pformat(locals()))

    output_path = Path(root, embedding_name, dataset_name)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline = FlairSequenceLabelingPreprocessingPipeline(
        dataset_name=dataset_name,
        input_column_name=input_column_name,
        target_column_name=target_column_name,
        persist_path=str(output_path),
    )
    result = pipeline.run()

    typer.echo(pprint.pformat(result))


typer.run(run)
