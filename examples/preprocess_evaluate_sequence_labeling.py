import pprint
import tempfile
from pathlib import Path
from typing import Optional

import flair
import torch
import typer

from embeddings.defaults import DATASET_PATH, RESULTS_PATH
from embeddings.pipeline.evaluation_pipeline import FlairSequenceLabelingEvaluationPipeline
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
    hidden_size: int = typer.Option(32, help="Number of hidden states in RNN."),
    evaluation_mode: str = typer.Option(
        "conll", help="Evaluation mode. Supported modes: [unit, conll, strict]."
    ),
    tagging_scheme: Optional[str] = typer.Option(
        None, help="Tagging scheme. Supported schemes: [IOB1, IOB2, IOE1, IOE2, IOBES, BILOU]"
    ),
) -> None:
    typer.echo(pprint.pformat(locals()))

    dataset_path = Path(DATASET_PATH, embedding_name, dataset_name)
    dataset_path.mkdir(parents=True, exist_ok=True)

    preprocessing_pipeline = FlairSequenceLabelingPreprocessingPipeline(
        dataset_name=dataset_name,
        input_column_name=input_column_name,
        target_column_name=target_column_name,
        persist_path=str(dataset_path),
        sample_missing_splits=True,
        ignore_test_subset=True,
    )
    preprocessing_pipeline.run()

    output_path = Path(RESULTS_PATH, embedding_name, dataset_name)
    output_path.mkdir(parents=True, exist_ok=True)
    persist_out_path = Path(output_path, f"{embedding_name}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluation_pipeline = FlairSequenceLabelingEvaluationPipeline(
        dataset_path=str(dataset_path),
        output_path=str(output_path),
        fine_tune_embeddings=False,
        hidden_size=hidden_size,
        embedding_name=embedding_name,
        persist_path=str(persist_out_path),
        predict_subset="dev",
        task_train_kwargs={"max_epochs": 1},
    )

    result = evaluation_pipeline.run()


typer.run(run)
