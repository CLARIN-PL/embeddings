from typing import Optional

import typer

from embeddings.config.lighting_config_space import LightingTextClassificationConfigSpace
from embeddings.defaults import RESULTS_PATH
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline
from embeddings.pipeline.lightning_hps_pipeline import OptimizedLightingClassificationPipeline
from embeddings.utils.loggers import LightningLoggingConfig
from embeddings.utils.utils import build_output_path, standardize_name

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
    n_trials: int = typer.Option(2, min=1, help="Number of search trials."),
    n_retrains: int = typer.Option(3, min=1, help="Number of search trials."),
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

    logging_config = LightningLoggingConfig.from_flags(
        wandb=wandb,
        tensorboard=tensorboard,
        csv=csv,
        tracking_project_name=tracking_project_name,
        wandb_entity=wandb_entity,
    )
    output_path = build_output_path(root, embedding_name_or_path, dataset_name)
    config_space = LightingTextClassificationConfigSpace(
        embedding_name_or_path=embedding_name_or_path,
    )
    pipeline = OptimizedLightingClassificationPipeline(
        config_space=config_space,
        dataset_name_or_path=dataset_name,
        input_column_name=input_column_name,
        target_column_name=target_column_name,
        logging_config=logging_config,
        n_trials=n_trials,
    ).persisting(
        best_params_path=output_path.joinpath("best_params.yaml"),
        log_path=output_path.joinpath("hps_log.pickle"),
        logging_config=logging_config,
    )
    df, metadata = pipeline.run(run_name=f"search-{run_name}")
    del pipeline

    for i in range(n_retrains):
        retrain_run_name = f"best-params-retrain-{run_name}-{i}"
        metadata["output_path"] = output_path / standardize_name(retrain_run_name)
        metadata["output_path"].mkdir()
        retrain_pipeline = LightningClassificationPipeline(
            logging_config=logging_config,
            **metadata,
        )
        retrain_pipeline.run(run_name=retrain_run_name)


typer.run(run)
