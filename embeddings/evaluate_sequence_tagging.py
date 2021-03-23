import argparse
import json
import os
from typing import Dict, List

import flair
import torch
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from sklearn.model_selection import train_test_split

from embeddings import datasets


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d',
        '--device',
        help='Device for Flair e.g. cpu',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-e',
        '--embedding',
        help='Huggingface embedding model name',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-u',
        '--url',
        help='Dataset url',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-r',
        '--root',
        help='Root directory for output files',
        type=str,
        required=False,
        default='resources/tagging/'
    )
    return parser.parse_args()


def setup(device):
    flair.device = torch.device(device)
    flair.set_seed(441)


def read_jsonl(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readlines()
        data = [json.loads(it) for it in data]
    return data


def split_train_test_dev(data):
    train, test = train_test_split(data, test_size=0.4, random_state=1)
    dev, test = train_test_split(test, test_size=0.5, random_state=1)
    return {
        'train': train,
        'dev': dev,
        'test': test
    }


def convert_jsonl_to_connl(data: List[Dict], out_path: str):
    for tweet in data:
        tokens = [it['text'] for it in tweet['tokens']]
        labels = ['O' for i in range(len(tokens))]
        for span in tweet['spans']:
            for idx in range(span['token_start'], span['token_end'] + 1):
                labels[idx] = span['label']

        with open(out_path, 'a') as f:
            for t, l in zip(tokens, labels):
                f.write(f'{t} \t  {l} \n')
            f.write('\n')


def evaluate_sequence_tagging():
    args = get_args()
    setup(args.device)

    out_dir = os.path.join(args.root, args.embedding)
    os.makedirs(out_dir, exist_ok=True)

    print('Downloading dataset...')
    downloaded = datasets.DatasetDownloader(
        root_dir=out_dir,
        url=args.url
    ).download()

    print('Preprocessing dataset...')
    if isinstance(downloaded, list):
        raise NotImplementedError

    data = read_jsonl(downloaded)
    splitted_data = split_train_test_dev(data)

    for key, ds in splitted_data.items():
        convert_jsonl_to_connl(
            ds, out_path=os.path.join(out_dir, f'{key}.csv')
        )

    columns = {0: 'text', 1: 'tag'}
    corpus = ColumnCorpus(
        out_dir,
        column_format=columns,
        train_file=os.path.join('train.csv'),
        test_file=os.path.join('test.csv'),
        dev_file=os.path.join('dev.csv')
    )
    tag_dictionary = corpus.make_tag_dictionary(tag_type='tag')

    print('Loading embeddings...')
    embeddings = TransformerWordEmbeddings(args.embedding)

    print('Training model...')
    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type='tag',
        use_crf=True
    )

    trainer = ModelTrainer(tagger, corpus)
    training_log = trainer.train(
        os.path.join(out_dir, 'tagger/'),
        learning_rate=0.1,
        mini_batch_size=64,
        max_epochs=150
    )
    print('Training log', training_log)
    print('Done!')

if __name__ == '__main__':
    evaluate_sequence_tagging()
