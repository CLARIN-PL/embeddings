import os
import tempfile

from datasets import load_dataset
from flair.data import Corpus
from flair.datasets import CSVClassificationDataset
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

MODEL = "/home/albert/dev/clarin/long-former-polish"

polemo_dataset = load_dataset("polemo2")

with tempfile.TemporaryDirectory() as tmp_dir_path:
    flair_datasets = {}

    for set_name in ["train", "validation", "test"]:
        label_map = polemo_dataset[set_name].features["target"].names
        polemo_dataset[set_name] = polemo_dataset[set_name].map(
            lambda row: {"named_target": label_map[row["target"]]},
            remove_columns=["target"],
        )
        polemo_dataset[set_name].to_csv(
            os.path.join(tmp_dir_path, f"{set_name}.csv"), header=False, index=False
        )

        column_name_map = {
            polemo_dataset[set_name].column_names.index("sentence"): "text",
            polemo_dataset[set_name].column_names.index("named_target"): "label",
        }

        flair_datasets[set_name] = CSVClassificationDataset(
            os.path.join(tmp_dir_path, f"{set_name}.csv"), column_name_map
        )

corpus = Corpus(
    train=flair_datasets["train"],
    dev=flair_datasets["validation"],
    test=flair_datasets["test"],
    name="polemo2",
)

embeddings = TransformerDocumentEmbeddings(MODEL, fine_tune=False)

label_dict = corpus.make_label_dictionary()
model = TextClassifier(embeddings, label_dictionary=label_dict)
trainer = ModelTrainer(model, corpus, use_tensorboard=True)
log = trainer.train(
    os.path.join("log", "classification"),
    # learning_rate=learning_rate,
    mini_batch_size=128,
    # max_epochs=max_epochs,
)
model.evaluate(corpus.test)
