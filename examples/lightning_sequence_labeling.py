import pprint
from pathlib import Path

from embeddings.defaults import RESULTS_PATH
from embeddings.pipeline.lightning.sequence_labeling import LightningSequenceLabelingPipeline

root: Path = RESULTS_PATH.joinpath("lightning_sequence_labeling")
root.mkdir(parents=True, exist_ok=True)

pipeline = LightningSequenceLabelingPipeline(
    embedding_name="allegro/herbert-base-cased",
    dataset_name="clarin-pl/kpwr-ner",
    input_column_name="tokens",
    target_column_name="ner",
    output_path=root,
    task_train_kwargs={"max_epochs": 1, "gpus": 1},
    task_model_kwargs={"pool_strategy": "cls", "learning_rate": 1e-4},
)
result = pipeline.run()
pprint.pformat(result)
