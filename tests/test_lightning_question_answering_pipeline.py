from pathlib import Path
from typing import Any, Dict

import datasets
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from _pytest.tmpdir import TempdirFactory
from numpy import array

from embeddings.config.lightning_config import LightningAdvancedConfig
from embeddings.evaluator.evaluation_results import QuestionAnsweringEvaluationResults
from embeddings.pipeline.hf_preprocessing_pipeline import HuggingFacePreprocessingPipeline
from embeddings.pipeline.lightning_question_answering import LightningQuestionAnsweringPipeline
from embeddings.task.lightning_task.question_answering import QuestionAnsweringTask


@pytest.fixture(scope="module")
def tmp_path_module(tmpdir_factory: TempdirFactory) -> Path:
    path = tmpdir_factory.mktemp(__name__)
    return Path(path)


@pytest.fixture(scope="module")
def dataset_dict() -> datasets.DatasetDict:
    train_dict = {
        "id": [
            "56be85543aeaaa14008c9063",
            "56be85543aeaaa14008c9065",
            "56be85543aeaaa14008c9066",
            "56bf6b0f3aeaaa14008c9601",
        ],
        "title": ["Beyoncé", "Beyoncé", "Beyoncé", "Beyoncé"],
        "context": [
            'Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".',
            'Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".',
            'Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".',
            'Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".',
        ],
        "question": [
            "When did Beyonce start becoming popular?",
            "What areas did Beyonce compete in when she was growing up?",
            "When did Beyonce leave Destiny's Child and become a solo singer?",
            "In what city and state did Beyonce  grow up? ",
        ],
        "answers": [
            {"text": ["in the late 1990s"], "answer_start": [269]},
            {"text": ["singing and dancing"], "answer_start": [207]},
            {"text": ["2003"], "answer_start": [526]},
            {"text": ["Houston, Texas"], "answer_start": [166]},
        ],
    }
    validation_dict = {
        "id": [
            "56ddde6b9a695914005b9628",
            "56ddde6b9a695914005b9629",
            "56ddde6b9a695914005b962a",
            "56ddde6b9a695914005b962b",
        ],
        "title": ["Normans", "Normans", "Normans", "Normans"],
        "context": [
            'The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.',
            'The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.',
            'The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.',
            'The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.',
        ],
        "question": [
            "In what country is Normandy located?",
            "When were the Normans in Normandy?",
            "From which countries did the Norse originate?",
            "Who was the Norse leader?",
        ],
        "answers": [
            {
                "text": ["France", "France", "France", "France"],
                "answer_start": [159, 159, 159, 159],
            },
            {
                "text": [
                    "10th and 11th centuries",
                    "in the 10th and 11th centuries",
                    "10th and 11th centuries",
                    "10th and 11th centuries",
                ],
                "answer_start": [94, 87, 94, 94],
            },
            {
                "text": [
                    "Denmark, Iceland and Norway",
                    "Denmark, Iceland and Norway",
                    "Denmark, Iceland and Norway",
                    "Denmark, Iceland and Norway",
                ],
                "answer_start": [256, 256, 256, 256],
            },
            {"text": ["Rollo", "Rollo", "Rollo", "Rollo"], "answer_start": [308, 308, 308, 308]},
        ],
    }
    train_dataset = datasets.Dataset.from_dict(train_dict)
    validation_dataset = datasets.Dataset.from_dict(validation_dict)
    return datasets.DatasetDict(
        {
            "train": train_dataset,
            "validation": validation_dataset,
        }
    )


@pytest.fixture(scope="module")
def config() -> LightningAdvancedConfig:
    return LightningAdvancedConfig(
        finetune_last_n_layers=0,
        task_train_kwargs={
            "max_epochs": 1,
            "devices": "auto",
            "accelerator": "auto",
        },
        task_model_kwargs={
            "learning_rate": 5e-4,
            "train_batch_size": 5,
            "eval_batch_size": 5,
            "use_scheduler": False,
            "optimizer": "AdamW",
            "adam_epsilon": 1e-8,
            "warmup_steps": None,
            "weight_decay": 1e-3,
            "max_seq_length": 512,
            "doc_stride": 128,
        },
        datamodule_kwargs={
            "max_seq_length": 64,
        },
        early_stopping_kwargs={
            "monitor": "val/Loss",
            "mode": "min",
            "patience": 3,
        },
        tokenizer_kwargs={},
        batch_encoding_kwargs={},
        dataloader_kwargs={},
        model_config_kwargs={},
    )


@pytest.fixture(scope="module")
def lightning_question_answering_pipeline(
    config: LightningAdvancedConfig, tmp_path_module: Path, dataset_dict: datasets.DatasetDict
):
    dataset = dataset_dict
    dataset.save_to_disk(tmp_path_module / "data_sample")
    return LightningQuestionAnsweringPipeline(
        embedding_name_or_path="xlm-roberta-base",
        output_path=tmp_path_module,
        config=config,
        evaluation_filename="evaluation.json",
        predict_subset="validation",
        model_checkpoint_kwargs={"filename": "last", "monitor": None, "save_last": False},
        dataset_name_or_path=tmp_path_module / "data_sample",
    )


def test_lightning_advanced_config(config):
    lightning_config = config
    assert isinstance(lightning_config, LightningAdvancedConfig)
    assert hasattr(lightning_config, "finetune_last_n_layers")
    assert hasattr(lightning_config, "task_train_kwargs")
    assert hasattr(lightning_config, "task_model_kwargs")
    assert hasattr(lightning_config, "datamodule_kwargs")
    assert hasattr(lightning_config, "early_stopping_kwargs")
    assert hasattr(lightning_config, "tokenizer_kwargs")
    assert hasattr(lightning_config, "batch_encoding_kwargs")
    assert hasattr(lightning_config, "dataloader_kwargs")
    assert hasattr(lightning_config, "model_config_kwargs")
    assert isinstance(lightning_config.task_model_kwargs, dict)
    assert "learning_rate" in lightning_config.task_model_kwargs.keys()


def test_lightning_question_answering_pipeline(
    lightning_question_answering_pipeline: LightningQuestionAnsweringPipeline,
):
    pipeline = lightning_question_answering_pipeline
    results = pipeline.run()
    assert isinstance(results, tuple)

    metrics = results[0]["validation"]
    assert 0 <= metrics["f1"] <= 100
    assert 0 <= metrics["total"] <= 100
    assert 0 <= metrics["HasAns_f1"] <= 100
    assert 0 <= metrics["HasAns_total"] <= 100
    assert 0 <= metrics["best_f1"] <= 100
    assert 0 <= metrics["HasAns_f1"] <= 100
