import pprint
from pathlib import Path

from pytorch_lightning.callbacks import EarlyStopping

from embeddings.defaults import RESULTS_PATH
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline

root: Path = RESULTS_PATH.joinpath("lightning_sequence_classification")
root.mkdir(parents=True, exist_ok=True)


pipeline = LightningClassificationPipeline(
    embedding_name="allegro/herbert-base-cased",
    dataset_name="clarin-pl/polemo2-official",
    input_column_name=["text"],
    target_column_name="target",
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
            EarlyStopping(monitor="val/F1", verbose=True, patience=5, min_delta=0.01, mode="max")
        ],
    },
    task_model_kwargs={"learning_rate": 5e-4, "use_scheduler": False},
)

result = pipeline.run()
pprint.pformat(result)
