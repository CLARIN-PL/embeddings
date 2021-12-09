import pprint
from pathlib import Path

import typer
from pytorch_lightning.callbacks import EarlyStopping

from embeddings.defaults import RESULTS_PATH
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline


def run(
    embedding_name: str = typer.Option(
        "allegro/herbert-base-cased", help="Hugging Face embedding model name or path."
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
) -> None:
    typer.echo(pprint.pformat(locals()))

    output_path = Path(root, embedding_name, dataset_name)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline = LightningClassificationPipeline(
        embedding_name=embedding_name,
        dataset_name=dataset_name,
        input_column_name=input_columns_name,
        target_column_name=target_column_name,
        output_path=root,
        load_dataset_kwargs={
            "train_domains": ["hotels", "medicine"],
            "dev_domains": ["hotels", "medicine"],
            "test_domains": ["hotels", "medicine"],
            "text_cfg": "text",
        },
        task_train_kwargs={
            "max_epochs": 10,
            "gpus": 1,
            "callbacks": [
                EarlyStopping(
                    monitor="val/F1", verbose=True, patience=5, min_delta=0.01, mode="max"
                )
            ],
        },
        task_model_kwargs={"learning_rate": 5e-4, "use_scheduler": False},
    )

    result = pipeline.run()
    typer.echo(pprint.pformat(result))


typer.run(run)
