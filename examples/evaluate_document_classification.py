import pprint

import typer

from embeddings.defaults import RESULTS_PATH
from embeddings.pipeline.flair_classification import FlairClassificationPipeline
from embeddings.utils.utils import build_output_path, format_eval_result

app = typer.Typer()


def run(
    embedding_name_or_path: str = typer.Option(
        "clarin-pl/word2vec-kgr10", help="Hugging Face embedding model name or path."
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

    output_path = build_output_path(root, embedding_name_or_path, dataset_name)
    pipeline = FlairClassificationPipeline(
        embedding_name=embedding_name_or_path,
        dataset_name=dataset_name,
        input_column_name=input_column_name,
        target_column_name=target_column_name,
        output_path=output_path,
    )
    result = pipeline.run()
    typer.echo(format_eval_result(result))


typer.run(run)
