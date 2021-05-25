from pathlib import Path
from typing import Union

import click
import tqdm
from sentence_transformers import SentenceTransformer, datasets, losses, models
from torch.utils.data import DataLoader


@click.command()
@click.option("--texts_file", type=str)
@click.option(
    "--model_name", type=str, default="sentence-transformers/paraphrase-MiniLM-L6-v2"
)  # sentence-transformers/paraphrase-xlm-r-multilingual-v1
@click.option("--batch_size", type=int, default=64)
@click.option("--epochs", type=int, default=1)
@click.option("--model_output_path", type=str, default=Path("pretrained_models"))
@click.option("--start_sentence", type=int, default=0)
@click.option("--end_sentence", type=int, default=5_000_000)
def main(
    texts_file,
    model_name: str,
    batch_size: int,
    epochs: int,
    model_output_path: Union[str, Path],
    start_sentence: int,
    end_sentence: int,
):
    train_sentences = []

    n_sentences = 0

    with open(texts_file, encoding="utf8") as f:
        for line in tqdm.tqdm(f, desc="read file..."):
            line = line.strip()
            # TODO: please verify min and max len of sentences
            if (
                len(line) >= 10
                and len(line.split()) <= 200
                and n_sentences >= start_sentence
            ):
                train_sentences.append(line)

            if n_sentences >= end_sentence:
                break

            n_sentences += 1

    if n_sentences <= end_sentence:
        end_sentence = n_sentences

    model_output_path = (
        model_output_path / f"{Path(texts_file).stem}-{str(end_sentence)}" / model_name
    )

    click.echo("{} train sentences".format(len(train_sentences)))

    word_embedding_model = models.Transformer(model_name)
    # Apply **cls** pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=False,
        pooling_mode_cls_token=True,
        pooling_mode_max_tokens=False,
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # We wrap our training sentences in the DenoisingAutoEncoderDataset to add deletion noise on the fly
    train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    train_loss = losses.DenoisingAutoEncoderLoss(
        model, decoder_name_or_path=model_name, tie_encoder_decoder=True
    )

    click.echo("Start training")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        weight_decay=0,
        scheduler="constantlr",
        optimizer_params={"lr": 3e-5},
        show_progress_bar=True,
        use_amp=True,  # Set to True, if your GPU supports FP16 cores
    )

    model_path = (model_output_path / "tsdae-model").as_posix()
    model.save(model_path)

    click.echo(f"Model has been saved into: {model_path}")


if __name__ == "__main__":
    main()
