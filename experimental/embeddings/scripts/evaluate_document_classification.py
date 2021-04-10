import os

from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

from experimental.datasets.hugging_face_dataset import HuggingFaceClassificationDataset

MODEL = "allegro/herbert-base-cased"

DATASET_NAME = "polemo2"
TARGET_COLUMN_NAME = "target"
INPUT_TEXT_COLUMN_NAME = "sentence"

dataset = HuggingFaceClassificationDataset(
    dataset_name=DATASET_NAME,
    input_text_column_name=INPUT_TEXT_COLUMN_NAME,
    target_column_name=TARGET_COLUMN_NAME,
)

flair_dataset = dataset.to_flair_column_corpus()

embeddings = TransformerDocumentEmbeddings(MODEL, fine_tune=False)

label_dict = flair_dataset.make_label_dictionary()
model = TextClassifier(embeddings, label_dictionary=label_dict)
trainer = ModelTrainer(model, flair_dataset, use_tensorboard=True)
log = trainer.train(
    os.path.join("log", "classification"),
    mini_batch_size=128,
)
result, loss = model.evaluate(flair_dataset.dev)

print(result.log_line)
print(result.detailed_results)
