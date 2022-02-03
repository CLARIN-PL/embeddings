import pprint
from typing import Tuple

import typer

from embeddings.defaults import RESULTS_PATH
from embeddings.pipeline.flair_pair_classification import FlairPairClassificationPipeline
from embeddings.utils.utils import build_output_path, format_eval_result

app = typer.Typer()


def run(
    model_name_or_path: str = typer.Option(
        "clarin-pl/word2vec-kgr10", help="Hugging Face embedding model name or path."
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

    output_path = build_output_path(root, model_name_or_path, dataset_name)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline = FlairPairClassificationPipeline(
        model_name_or_path, dataset_name, input_columns_names_pair, target_column_name, output_path
    )
    result = pipeline.run()
    typer.echo(format_eval_result(result))


typer.run(run)
