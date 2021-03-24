import argparse
import os
import pickle

import flair
import torch
from flair.embeddings import TransformerWordEmbeddings

from experimental.embeddings import datasets
from experimental.embeddings.tasks import sequence_tagging as st


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--device",
        help="Device for Flair e.g. cpu",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-e",
        "--embedding",
        help="Huggingface embedding model name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        help="Dataset name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset_url",
        help="Dataset download url",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-r",
        "--root",
        help="Root directory for output files",
        type=str,
        required=False,
        default="resources/tagging/",
    )
    return parser.parse_args()


def setup(device: str) -> None:
    flair.device = torch.device(device)  # type:ignore
    flair.set_seed(441)


def evaluate_sequence_tagging() -> None:
    args = get_args()
    setup(args.device)

    out_dir = os.path.join(args.root, args.embedding)
    os.makedirs(out_dir, exist_ok=True)

    ds_args = {"url": args.dataset_url} if args.dataset_url else {}
    dataset_cls = datasets.get_dataset_cls(args.dataset)
    dataset = dataset_cls(**ds_args)
    corpus = dataset.to_flair_column_corpus()
    tag_dictionary = corpus.make_tag_dictionary(tag_type="tag")

    print("Loading embeddings...")
    embeddings = TransformerWordEmbeddings(args.embedding)

    print("Training model...")
    tagger = st.FlairSequenceTagger(
        embeddings=embeddings,
        hidden_dim=256,
        tag_dictionary=tag_dictionary,
        output_path=out_dir,
    )

    training_log = tagger.fit(corpus)
    with open(os.path.join(out_dir, "results.pkl"), "wb") as f:
        pickle.dump(obj=training_log, file=f)

    print("Done!", training_log)


if __name__ == "__main__":
    evaluate_sequence_tagging()
