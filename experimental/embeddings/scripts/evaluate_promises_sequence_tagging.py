# from pathlib import Path
# from typing import Optional

# import flair
# import torch
# import typer
# from flair.embeddings import TransformerWordEmbeddings

# import experimental.datasets.utils.misc
# from embeddings.defaults import RESULTS_PATH

# app = typer.Typer()


# def setup(device: str) -> None:
#     flair.device = torch.device(device)  # type:ignore
#     flair.set_seed(441)


# def evaluate_sequence_tagging(
#     embedding: str = typer.Option(..., help="Huggingface embedding model name"),
#     dataset_name: str = typer.Option(..., help="Dataset name"),
#     dataset_url: Optional[str] = typer.Option(None, help="Dataset download url"),
#     root: str = typer.Option(
#         RESULTS_PATH.joinpath("tagging"),
#         help="Root directory for output files",
#     ),
#     device: str = typer.Option("cpu", help="Device for Flair e.g. cpu"),
# ) -> None:
#     setup(device)

#     out_dir = Path(root, embedding, dataset_name)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     ds_args = {"url": dataset_url} if dataset_url else {}
#     dataset_cls = experimental.datasets.utils.misc.get_dataset_cls(dataset_name)
#     dataset = dataset_cls(**ds_args)
#     corpus = dataset.to_flair_column_corpus()
#     tag_dictionary = corpus.make_tag_dictionary(tag_type="tag")

#     test_sentences = corpus.test.sentences.copy()

#     print("Loading embeddings...")
#     embeddings = TransformerWordEmbeddings(embedding)

#     print("Training model...")
#     tagger = FlairSequenceTagger(
#         embeddings=embeddings,
#         hidden_dim=256,
#         tag_dictionary=tag_dictionary,
#         output_path=str(out_dir),
#     )

#     training_log = tagger.fit(corpus)
#     evaluation_log = tagger.evaluate(test_sentences)
#     with out_dir.joinpath("results.pkl").open(mode="wb") as f:
#         pickle.dump(obj={"training_log": training_log, "evaluation_log": evaluation_log}, file=f)

#     print("Done!", training_log)


# typer.run(evaluate_sequence_tagging)
