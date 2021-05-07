from pathlib import Path
from pprint import pprint

import typer
from flair.embeddings import TransformerDocumentEmbeddings

from experimental.datasets.hugging_face_dataset import HuggingFaceClassificationDataset
from experimental.defaults import RESULTS_PATH
from experimental.embeddings.tasks.text_classification import FlairTextClassification

app = typer.Typer()


def run(
    embedding: str = typer.Option(..., help="Hugging Face embedding model name"),
    dataset_name: str = typer.Option(...),
    input_text_column_name: str = typer.Option("sentence"),
    target_column_name: str = typer.Option("target"),
    root: str = typer.Option(RESULTS_PATH.joinpath("document_classification")),
) -> None:
    pprint(locals())
    out_dir = Path(root, embedding, dataset_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = HuggingFaceClassificationDataset(
        dataset_name=dataset_name,
        input_text_column_name=input_text_column_name,
        target_column_name=target_column_name,
    )

    corpus = dataset.to_flair_column_corpus()
    label_dict = corpus.make_label_dictionary()

    embeddings = TransformerDocumentEmbeddings(embedding, fine_tune=False)
    classifier = FlairTextClassification(embeddings, label_dict, out_dir)
    classifier.fit(corpus, max_epochs=100)

    result = classifier.evaluate(corpus.test)
    pprint(result)


typer.run(run)
