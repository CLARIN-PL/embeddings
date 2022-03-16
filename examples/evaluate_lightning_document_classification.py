import pprint
from typing import Optional

import typer

from embeddings.defaults import RESULTS_PATH
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline
from embeddings.utils.utils import build_output_path, format_eval_result

LightningClassificationPipeline.DEFAULT_DATAMODULE_KWARGS.update(
    {
        "downsample_train": 0.01,
        "downsample_val": 0.04,
        "downsample_test": 0.04,
    }
)


def run(
    embedding_name_or_path: str = typer.Option(
        "hf-internal-testing/tiny-albert", help="Hugging Face embedding model name or path."
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
    run_name: Optional[str] = typer.Option(None, help="Name of run used for logging."),
    wandb: bool = typer.Option(False, help="Flag for using wandb."),
    tensorboard: bool = typer.Option(False, help="Flag for using tensorboard."),
    csv: bool = typer.Option(False, help="Flag for using csv."),
    wandb_project: Optional[str] = typer.Option(None, help="Name of wandb project."),
    wandb_entity: Optional[str] = typer.Option(None, help="Name of entity project"),
) -> None:
    typer.echo(pprint.pformat(locals()))

    output_path = build_output_path(root, embedding_name_or_path, dataset_name)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline = LightningClassificationPipeline(
        embedding_name_or_path=embedding_name_or_path,
        dataset_name_or_path=dataset_name,
        input_column_name=input_columns_name,
        target_column_name=target_column_name,
        # finetune_last_n_layers=4,
        task_train_kwargs={"max_epochs": 50},
        output_path=output_path,
        logging_kwargs={
            "use_tensorboard": tensorboard,
            "use_wandb": wandb,
            "use_csv": csv,
            "wandb_project": wandb_project,
            "wandb_entity": wandb_entity,
        },
    )

    result = pipeline.run(run_name=run_name)
    typer.echo(format_eval_result(result))


typer.run(run)
