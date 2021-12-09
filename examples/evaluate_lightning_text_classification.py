import pprint
from pathlib import Path

from embeddings.defaults import RESULTS_PATH
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline

root: Path = RESULTS_PATH.joinpath("lightning_sequence_classification")
root.mkdir(parents=True, exist_ok=True)

pipeline = LightningClassificationPipeline(
    embedding_name="allegro/herbert-base-cased",
    dataset_name="clarin-pl/cst-wikinews",
    input_column_name=["sentence_1", "sentence_2"],
    output_path=root,
    target_column_name="label",
    task_train_kwargs={
        "max_epochs": 1,
        "gpus": 1,
        "limit_val_batches": 0,
        "num_sanity_val_steps": 0,
    },
    task_model_kwargs={"pool_strategy": "cls", "learning_rate": 5e-4},
)

# pipeline = TextClassificationPipeline(
#     embedding_name="allegro/herbert-base-cased",
#     dataset_name="clarin-pl/polemo2-official",
#     input_column_name=["text"],
#     target_column_name="target",
#     output_path=root,
#     load_dataset_kwargs={
#         "train_domains": ["hotels", "medicine"],
#         "dev_domains": ["hotels", "medicine"],
#         "test_domains": ["hotels", "medicine"],
#         "text_cfg": "text",
#     },
#     task_train_kwargs={"max_epochs": 2, "gpus": 1},
#     task_model_kwargs={"learning_rate": 5e-4, "use_scheduler": True},
# )

result = pipeline.run()
pprint.pformat(result)
