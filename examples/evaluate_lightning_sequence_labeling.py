import pprint
from typing import Optional

import typer

from embeddings.defaults import RESULTS_PATH
from embeddings.evaluator.sequence_labeling_evaluator import EvaluationMode, TaggingScheme
from embeddings.pipeline.lightning_sequence_labeling import LightningSequenceLabelingPipeline
from embeddings.utils.utils import build_output_path, format_eval_result


def run(
    embedding_name_or_path: str = typer.Option(
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
    evaluation_mode: EvaluationMode = typer.Option(
        EvaluationMode.CONLL,
        help="Evaluation mode. Supported modes: [unit, conll, strict].",
    ),
    tagging_scheme: Optional[TaggingScheme] = typer.Option(
        None, help="Tagging scheme. Supported schemes: [IOB1, IOB2, IOE1, IOE2, IOBES, BILOU]"
    ),
    root: str = typer.Option(RESULTS_PATH.joinpath("lightning_sequence_classification")),
    run_name: Optional[str] = typer.Option(None, help="Name of run used for logging."),
    wandb: bool = typer.Option(False, help="Flag for using wandb."),
    tensorboard: bool = typer.Option(False, help="Flag for using tensorboard."),
    csv: bool = typer.Option(False, help="Flag for using csv."),
    tracking_project_name: Optional[str] = typer.Option(None, help="Name of wandb project."),
    wandb_entity: Optional[str] = typer.Option(None, help="Name of wandb entity."),
) -> None:
    typer.echo(pprint.pformat(locals()))

    output_path = build_output_path(root, embedding_name_or_path, dataset_name)
    pipeline = LightningSequenceLabelingPipeline(
        embedding_name_or_path=embedding_name_or_path,
        dataset_name_or_path=dataset_name,
        input_column_name=input_column_name,
        target_column_name=target_column_name,
        output_path=root,
        evaluation_mode=evaluation_mode,
        tagging_scheme=tagging_scheme,
        logging_kwargs={
            "use_tensorboard": tensorboard,
            "use_wandb": wandb,
            "use_csv": csv,
            "tracking_project_name": tracking_project_name,
            "wandb_entity": wandb_entity,
        },
    )

    result = pipeline.run()
    typer.echo(format_eval_result(result))


typer.run(run)
