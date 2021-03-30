import pickle
import pprint
from collections import namedtuple
from os.path import basename, join
from pathlib import Path
from typing import Iterable

import click
from aspect_extraction import AspectExtraction
from keras.callbacks import TensorBoard, EarlyStopping
from keras.utils import to_categorical
from nlp_architect.contrib.keras.callbacks import ConllCallback
from nlp_architect.data.sequential_tagging import SequentialTaggingDataset
from nlp_architect.utils.metrics import get_conll_scores
from tqdm import tqdm

DatasetFiles = namedtuple("Dataset", ["name", "train_file", "test_file"])

EMBEDDINGS = [
    # Google's word2vec news
    ("GoogleNews-vectors-negative300.txt", 300),
    # https://nlp.stanford.edu/projects/glove/
    ("glove.840B.300d.txt", 300),
    ("glove.42B.300d.txt", 300),
    ("glove.6B.300d.txt", 300),
    ("glove.6B.200d.txt", 200),
    ("glove.6B.100d.txt", 100),
    ("glove.6B.50d.txt", 50),
    ("glove.twitter.27B.25d.txt", 25),
    ("glove.twitter.27B.50d.txt", 50),
    ("glove.twitter.27B.100d.txt", 100),
    ("glove.twitter.27B.200d.txt", 200),
    ("numberbatch-en.txt", 300),
    ("crawl-300d-2M.vec", 300),
    ("wiki-news-300d-1M-subword.vec", 300),
    ("wiki-news-300d-1M.vec", 300),
    # https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/
    ("bow2.words", 300),
    ("bow2.contexts", 300),
    ("bow5.words", 300),
    ("bow5.contexts", 300),
    ("deps.words", 300),
    ("deps.contexts", 300),
    # http://www.ims.uni-stuttgart.de/forschung/ressourcen/experiment-daten/sota-sentiment.html
    ("sota-google.txt", 300),
    ("sota-retrofit-600.txt", 600),
    ("sota-sswe-50.txt", 50),
    ("sota-wiki-600.txt", 600),
    # Cambria CNN aspects based on Amazon reviews
    ("sentic2vec.txt", 300),
    ("lexvec.commoncrawl.ngramsubwords.300d.W.pos.vectors", 300),
    ("lexvec.enwiki+newscrawl.300d.W.pos.vectors", 300),
]

DATASETS_PATS = ["semeval/2014", "semeval/2014/poria"]
TF = [True, False]


@click.command()
@click.option(
    "--embeddings-path", required=True, type=Path, help="Path to the embeddings"
)
@click.option(
    "--logs-path",
    required=True,
    type=Path,
    help="Path to output logs and tensorboard objects",
)
@click.option(
    "--models-path",
    required=False,
    default=None,
    type=Path,
    help="Path to output models",
)
@click.option(
    "--epochs", required=False, default=25, help="Number of epochs to calculate"
)
@click.option(
    "--tag-number",
    required=False,
    default=3,
    help="Number of column with tag to classify",
)
def run_evaluation_multi_datasets_and_multi_embeddings(
    embeddings_path: Path,
    models_path: Path,
    logs_path: Path,
    epochs: int,
    tag_number: int,
):
    for embedding, word_embedding_dims in tqdm(EMBEDDINGS, desc="Embeddings progress"):
        for word_embedding_flag in [True]:
            for char_embedding_flag in TF:
                for bilstm_layer in TF:
                    for crf_layer in TF:

                        # we can't process without vectorization
                        if not word_embedding_flag and not char_embedding_flag:
                            continue

                        click.echo("Embedding: " + embedding)
                        embedding_model = embeddings_path / embedding
                        embedding_name = Path(embedding).stem
                        if models_path is None:
                            models_output = Path("models") / embedding_name
                        else:
                            models_output = (
                                models_path / Path("models") / embedding_name
                            )
                        models_output.mkdir(parents=True, exist_ok=True)

                        word_embedding_dims = (
                            word_embedding_dims if word_embedding_flag else 0
                        )
                        character_embedding_dims = 25 if char_embedding_flag else 0

                        for dataset_file in tqdm(
                            get_aspect_datasets(), desc="Datasets progress"
                        ):
                            click.echo("Dataset: " + dataset_file.train_file.as_posix())
                            run_aspect_sequence_tagging(
                                train_file=dataset_file.train_file.as_posix(),
                                test_file=dataset_file.test_file.as_posix(),
                                embedding_model=embedding_model.as_posix(),
                                models_path=models_output.as_posix(),
                                logs_path=logs_path,
                                tag_num=tag_number,
                                epoch=epochs,
                                dropout=0.5,
                                character_embedding_dims=character_embedding_dims,
                                char_features_lstm_dims=character_embedding_dims,
                                word_embedding_dims=word_embedding_dims,
                                entity_tagger_lstm_dims=word_embedding_dims
                                + character_embedding_dims,
                                tagger_fc_dims=word_embedding_dims
                                + character_embedding_dims,
                                augment_data=False,
                                bilstm_layer=bilstm_layer,
                                crf_layer=crf_layer,
                                word_embedding_flag=word_embedding_flag,
                                char_embedding_flag=char_embedding_flag,
                            )


def get_aspect_datasets() -> Iterable[DatasetFiles]:
    datasets = []
    for datasets_path in tqdm(DATASETS_PATS, desc="Datasets"):
        datasets_path = Path(datasets_path)
        train_files = list(datasets_path.glob("*train.conll"))
        test_files = list(datasets_path.glob("*test.conll"))
        for train_file in tqdm(train_files, desc="Datasets progress"):
            test_file = [
                f
                for f in test_files
                if train_file.stem.replace("train", "test") == f.stem
            ][0]
            dataset_name = test_file.stem.replace("-test", "")
            datasets.append(
                DatasetFiles(
                    name=dataset_name, train_file=train_file, test_file=test_file
                )
            )
    return datasets


def run_aspect_sequence_tagging(
    train_file,
    test_file,
    models_path: str,
    logs_path: Path,
    augment_data: bool,
    embedding_model: str,
    word_embedding_dims: int,
    character_embedding_dims: int,
    char_features_lstm_dims: int,
    entity_tagger_lstm_dims: int,
    tagger_fc_dims: int,
    batch_size=10,
    epoch=50,
    tag_num=2,
    sentence_length=30,
    word_length=20,
    dropout=0.2,
    bilstm_layer: bool = True,
    crf_layer: bool = False,
    word_embedding_flag: bool = True,
    char_embedding_flag: bool = True,
    similarity_threshold: float = 0.8,
):
    network_params = [
        ("char", char_embedding_flag),
        ("word", word_embedding_flag),
        ("bilstm", bilstm_layer),
        ("lstm", not bilstm_layer),
        ("crf", crf_layer),
        (str(epoch) + "epochs", True),
        (str(similarity_threshold) + "augmented", augment_data),
    ]

    network_params_string = "-".join([param for param, flag in network_params if flag])

    trained_models_path = Path("trained", models_path)
    trained_models_path.mkdir(exist_ok=True, parents=True)
    logs_path = logs_path / models_path
    logs_path.mkdir(exist_ok=True, parents=True)

    # load dataset and parameters
    model_name = (
        "model-info"
        + "-"
        + network_params_string
        + "-"
        + basename(train_file)
        + ".info"
    )
    models_path = join(models_path, model_name)
    if Path(models_path).exists():
        click.echo("Model has been already computed and saved!")
        return

    dataset = SequentialTaggingDataset(
        train_file=train_file,
        test_file=test_file,
        augment_data=augment_data,
        similarity_threshold=similarity_threshold,
        max_sentence_length=sentence_length,
        max_word_length=word_length,
        tag_field_no=tag_num,
    )

    # get the train and test data sets
    x_train, x_char_train, y_train = dataset.train
    x_test, x_char_test, y_test = dataset.test

    if word_embedding_flag and char_embedding_flag:
        x_train = [x_train, x_char_train]
        x_test = [x_test, x_char_test]
    elif word_embedding_flag and not char_embedding_flag:
        x_train = x_train
        x_test = x_test
    elif not word_embedding_flag and char_embedding_flag:
        x_train = x_char_train
        x_test = x_char_test
    else:
        raise Exception("Wrong features")

    num_y_labels = len(dataset.y_labels) + 1
    vocabulary_size = dataset.word_vocab_size + 1
    char_vocabulary_size = dataset.char_vocab_size + 1

    y_test = to_categorical(y_test, num_y_labels)
    y_train = to_categorical(y_train, num_y_labels)

    aspect_model = AspectExtraction()
    aspect_model.build(
        sentence_length,
        word_length,
        num_y_labels,
        dataset.word_vocab,
        vocabulary_size,
        char_vocabulary_size,
        word_embedding_dims=word_embedding_dims,
        char_embedding_dims=character_embedding_dims,
        word_lstm_dims=char_features_lstm_dims,
        tagger_lstm_dims=entity_tagger_lstm_dims,
        tagger_fc_dims=tagger_fc_dims,
        dropout=dropout,
        external_embedding_model=embedding_model,
        bilstm_layer=bilstm_layer,
        crf_layer=crf_layer,
        word_embedding_flag=word_embedding_flag,
        char_embedding_flag=char_embedding_flag,
    )

    # Set callback functions to early stop training and save the best model so far
    tensorboard_path = (logs_path / ("tensorboard-" + model_name)).as_posix()
    print("Tensorboard: " + tensorboard_path)

    callbacks = [
        ConllCallback(x_test, y_test, dataset.y_labels, batch_size=batch_size),
        TensorBoard(log_dir=tensorboard_path),
        EarlyStopping(monitor="val_loss", patience=2),
        # ModelCheckpoint(
        #     filepath=(trained_models_path / '{}-best_model.h5'.format(model_name)).as_posix(),
        #     monitor='val_loss',
        #     save_best_only=True
        # )
    ]

    aspect_model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epoch,
        callbacks=callbacks,
        validation=(x_test, y_test),
    )

    # running predictions
    predictions = aspect_model.predict(x=x_test, batch_size=1)
    eval = get_conll_scores(
        predictions, y_test, {v: k for k, v in dataset.y_labels.items()}
    )
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(eval)

    # saving model
    with open(models_path, "wb") as fp:
        info = {
            "sentence_len": sentence_length,
            "word_len": word_length,
            "num_of_labels": num_y_labels,
            "labels_id_to_word": {v: k for k, v in dataset.y_labels.items()},
            "epoch": epoch,
            # 'word_vocab': dataset.word_vocab,
            "vocab_size": vocabulary_size,
            "char_vocab_size": char_vocabulary_size,
            "char_vocab": dataset.char_vocab,
            "word_embedding_dims": word_embedding_dims,
            "char_embedding_dims": character_embedding_dims,
            "word_lstm_dims": char_features_lstm_dims,
            "tagger_lstm_dims": entity_tagger_lstm_dims,
            "dropout": dropout,
            "external_embedding_model": embedding_model,
            "train_file": train_file,
            "test_file": test_file,
            # 'test_raw_sentences': dataset.test_raw_sentences,
            "eval": eval,
            # 'data_augmentation': dataset.data_augmentation,
            # 'augment_data': augment_data,
            "similarity_threshold": similarity_threshold,
            "bilstm_layer": bilstm_layer,
            "crf_layer": crf_layer,
            "word_embedding_layer": word_embedding_flag,
            "char_embedding_layer": char_embedding_flag,
        }
        print("Save model in: " + models_path)
        pickle.dump(info, fp)


if __name__ == "__main__":
    run_evaluation_multi_datasets_and_multi_embeddings()
