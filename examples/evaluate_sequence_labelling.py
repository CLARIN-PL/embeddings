import pprint
from pathlib import Path
from typing import Optional

import typer

from embeddings.defaults import RESULTS_PATH
from embeddings.pipeline.hugging_face_sequence_labeling import HuggingFaceSequenceLabelingPipeline

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
    root: str = typer.Option(RESULTS_PATH.joinpath("pos_tagging")),
    hidden_size: int = typer.Option(32, help="Number of hidden states in RNN."),
    evaluation_mode: str = typer.Option(
        "conll", help="Evaluation mode. Supported modes: [unit, conll, strict]."
    ),
    tagging_scheme: Optional[str] = typer.Option(
        None, help="Tagging scheme. Supported schemes: [IOB1, IOB2, IOE1, IOE2, IOBES, BILOU]"
    ),
) -> None:
    typer.echo(pprint.pformat(locals()))
    output_path = Path(root, embedding_name, dataset_name)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline = HuggingFaceSequenceLabelingPipeline(
        embedding_name,
        dataset_name,
        input_column_name,
        target_column_name,
        output_path,
        hidden_size,
        evaluation_mode,
        tagging_scheme,
    )
    result = pipeline.run()
    typer.echo(pprint.pformat(result))


typer.run(run)
