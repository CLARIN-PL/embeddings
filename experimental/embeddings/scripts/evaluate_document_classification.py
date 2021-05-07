import os
from pprint import pprint

import typer
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

from experimental.datasets.hugging_face_dataset import HuggingFaceClassificationDataset

app = typer.Typer()


def run(
    model: str = typer.Option(...),
    dataset_name: str = typer.Option(...),
    input_text_column_name: str = typer.Option("sentence"),
    target_column_name: str = typer.Option("target"),
) -> None:
    pprint(locals())

    dataset = HuggingFaceClassificationDataset(
        dataset_name=dataset_name,
        input_text_column_name=input_text_column_name,
        target_column_name=target_column_name,
    )

    flair_dataset = dataset.to_flair_column_corpus()
    embeddings = TransformerDocumentEmbeddings(model, fine_tune=False)

    label_dict = flair_dataset.make_label_dictionary()
    classifier = TextClassifier(embeddings, label_dictionary=label_dict)
    trainer = ModelTrainer(classifier, flair_dataset, use_tensorboard=True)
    log = trainer.train(os.path.join("log", "classification"), mini_batch_size=128)

    print("TEST evaluation:")
    result, loss = classifier.evaluate(flair_dataset.test)
    print(result.log_line)
    print(result.detailed_results)


typer.run(run)
