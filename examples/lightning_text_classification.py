import pprint

from embeddings.pipeline.lightning_classification import TorchClassificationPipeline

# pipeline = TorchClassificationPipeline(
#     embedding_name="allegro/herbert-base-cased",
#     dataset_name="clarin-pl/cst-wikinews",
#     input_column_name=["sentence_1", "sentence_2"],
#     target_column_name="label",
#     task_train_kwargs={"max_epochs": 1, "gpus": 1},
#     task_model_kwargs={"pool_strategy": "cls", "learning_rate": 5e-4}
# )
pipeline = TorchClassificationPipeline(
    embedding_name="allegro/herbert-base-cased",
    dataset_name="clarin-pl/polemo2-official",
    input_column_name=["text"],
    target_column_name="target",
    load_dataset_kwargs={
        "train_domains": ["hotels", "medicine"],
        "dev_domains": ["hotels", "medicine"],
        "test_domains": ["hotels", "medicine"],
        "text_cfg": "text",
    },
    task_train_kwargs={"max_epochs": 10, "gpus": 1},
    task_model_kwargs={"pool_strategy": "cls", "learning_rate": 5e-4}
)
result = pipeline.run()
pprint.pformat(result)
