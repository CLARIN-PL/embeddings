import pprint
from typing import Optional

import typer

from embeddings.defaults import DATASET_PATH, RESULTS_PATH
from embeddings.pipeline.evaluation_pipeline import FlairSequenceLabelingEvaluationPipeline
from embeddings.pipeline.preprocessing_pipeline import FlairSequenceLabelingPreprocessingPipeline
from embeddings.utils.utils import build_output_path

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
    hidden_size: int = typer.Option(32, help="Number of hidden states in RNN."),
    evaluation_mode: str = typer.Option(
        "conll", help="Evaluation mode. Supported modes: [unit, conll, strict]."
    ),
    tagging_scheme: Optional[str] = typer.Option(
        None, help="Tagging scheme. Supported schemes: [IOB1, IOB2, IOE1, IOE2, IOBES, BILOU]"
    ),
) -> None:
    typer.echo(pprint.pformat(locals()))
    dataset_path = build_output_path(DATASET_PATH, embedding_name, dataset_name)
    dataset_path.mkdir(parents=True, exist_ok=True)

    preprocessing_pipeline = FlairSequenceLabelingPreprocessingPipeline(
        dataset_name=dataset_name,
        input_column_name=input_column_name,
        target_column_name=target_column_name,
        persist_path=str(dataset_path),
        sample_missing_splits=(0.1, 0.1),
        ignore_test_subset=False,
    )
    dataset = preprocessing_pipeline.run()

    output_path = build_output_path(RESULTS_PATH, embedding_name, dataset_name)
    output_path.mkdir(parents=True, exist_ok=True)
    persist_out_path = output_path.joinpath(f"{embedding_name}.json")
    persist_out_path.parent.mkdir(parents=True, exist_ok=True)

    evaluation_pipeline = FlairSequenceLabelingEvaluationPipeline(
        dataset_path=str(dataset_path),
        embedding_name=embedding_name,
        output_path=str(output_path),
        hidden_size=hidden_size,
        persist_path=str(persist_out_path),
        predict_subset="test",
        task_train_kwargs={"max_epochs": 1},
    )

    result = evaluation_pipeline.run()
    typer.echo(pprint.pformat(result))


typer.run(run)
