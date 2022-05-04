from typing import Optional

import typer

from embeddings.config.lighting_config_space import LightingSequenceLabelingConfigSpace
from embeddings.defaults import RESULTS_PATH
from embeddings.pipeline.lightning_hps_pipeline import OptimizedLightingSequenceLabelingPipeline
from embeddings.pipeline.lightning_sequence_labeling import LightningSequenceLabelingPipeline
from embeddings.utils.loggers import LightningLoggingConfig
from embeddings.utils.utils import build_output_path

app = typer.Typer()


def run(
    embedding_name_or_path: str = typer.Option(
        "allegro/herbert-base-cased", help="Hugging Face embedding model name or path."
    ),
    dataset_name: str = typer.Option(
        "clarin-pl/polemo2-official", help="Hugging Face dataset name or path."
    ),
    input_column_name: str = typer.Option(
        "text", help="Pair of column names that contain texts to classify."
    ),
    target_column_name: str = typer.Option(
        "target", help="Column name that contains label for classification."
    ),
    n_trials: int = typer.Option(2, help="Number of search trials."),
    root: str = typer.Option(RESULTS_PATH.joinpath("lightning_sequence_classification")),
    run_name: Optional[str] = typer.Option(None, help="Name of run used for logging."),
    wandb: bool = typer.Option(False, help="Flag for using wandb."),
    tensorboard: bool = typer.Option(False, help="Flag for using tensorboard."),
    csv: bool = typer.Option(False, help="Flag for using csv."),
    tracking_project_name: Optional[str] = typer.Option(None, help="Name of wandb project."),
    wandb_entity: Optional[str] = typer.Option(None, help="Name of entity project"),
) -> None:
    if not run_name:
        run_name = embedding_name_or_path
    output_path = build_output_path(root, embedding_name_or_path, dataset_name)
    config_space = LightingSequenceLabelingConfigSpace(
        embedding_name_or_path=embedding_name_or_path,
    )
    pipeline = OptimizedLightingSequenceLabelingPipeline(
        config_space=config_space,
        dataset_name_or_path=dataset_name,
        input_column_name=input_column_name,
        target_column_name=target_column_name,
        logging_config=LightningLoggingConfig.from_flags(
            wandb=wandb,
            tensorboard=tensorboard,
            csv=csv,
            tracking_project_name=tracking_project_name,
            wandb_entity=wandb_entity,
        ),
        n_trials=n_trials,
    ).persisting(
        best_params_path=output_path.joinpath("best_params.yaml"),
        log_path=output_path.joinpath("hps_log.pickle"),
    )
    df, metadata = pipeline.run(run_name=f"search-{run_name}")
    del pipeline

    metadata["output_path"] = output_path
    retrain_pipeline = LightningSequenceLabelingPipeline(
        logging_config=LightningLoggingConfig.from_flags(
            wandb=wandb,
            tensorboard=tensorboard,
            csv=csv,
            tracking_project_name=tracking_project_name,
            wandb_entity=wandb_entity,
        ),
        **metadata,
    )
    retrain_pipeline.run(run_name=f"best-params-retrain-{run_name}")


typer.run(run)
