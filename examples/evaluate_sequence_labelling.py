import pprint
from typing import Optional

import typer

from embeddings.defaults import RESULTS_PATH
from embeddings.evaluator.sequence_labeling_evaluator import SequenceLabelingEvaluator
from embeddings.pipeline.flair_sequence_labeling import FlairSequenceLabelingPipeline
from embeddings.utils.utils import build_output_path

app = typer.Typer()


def run(
    embedding_name_or_path: str = typer.Option(
        "clarin-pl/word2vec-kgr10", help="Hugging Face embedding model name or path."
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
    hidden_size: int = typer.Option(256, help="Number of hidden states in RNN."),
    evaluation_mode: SequenceLabelingEvaluator.EvaluationMode = typer.Option(
        SequenceLabelingEvaluator.EvaluationMode.CONLL,
        help="Evaluation mode. Supported modes: [unit, conll, strict].",
    ),
    tagging_scheme: Optional[SequenceLabelingEvaluator.TaggingScheme] = typer.Option(
        None, help="Tagging scheme. Supported schemes: [IOB1, IOB2, IOE1, IOE2, IOBES, BILOU]"
    ),
) -> None:
    typer.echo(pprint.pformat(locals()))

    output_path = build_output_path(root, embedding_name_or_path, dataset_name)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline = FlairSequenceLabelingPipeline(
        embedding_name=embedding_name_or_path,
        dataset_name=dataset_name,
        input_column_name=input_column_name,
        target_column_name=target_column_name,
        output_path=output_path,
        hidden_size=hidden_size,
        evaluation_mode=evaluation_mode,
        tagging_scheme=tagging_scheme,
    )
    result = pipeline.run()
    typer.echo(pprint.pformat(result))


typer.run(run)
