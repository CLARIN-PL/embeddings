import pprint
from typing import Optional

import typer

from embeddings.defaults import RESULTS_PATH
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline
from embeddings.utils.loggers import LightningLoggingConfig
from embeddings.utils.utils import build_output_path, format_eval_result


def run(
    embedding_name_or_path: str = typer.Option(
        "allegro/herbert-base-cased", help="Hugging Face embedding model name or path."
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
    pipeline = LightningClassificationPipeline(
        embedding_name_or_path=embedding_name_or_path,
        dataset_name_or_path=dataset_name,
        input_column_name=input_column_name,
        target_column_name=target_column_name,
        output_path=output_path,
        logging_config=LightningLoggingConfig.from_flags(
            wandb=wandb,
            tensorboard=tensorboard,
            csv=csv,
            tracking_project_name=tracking_project_name,
            wandb_entity=wandb_entity,
        ),
    )

    result = pipeline.run(run_name=run_name)
    typer.echo(format_eval_result(result))


typer.run(run)
