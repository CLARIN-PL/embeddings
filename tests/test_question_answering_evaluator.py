import random
from pathlib import Path
from typing import Any, Dict

import datasets
import numpy as np
import pytest
import torch
from _pytest.tmpdir import TempdirFactory

from embeddings.data.qa_datamodule import QuestionAnsweringDataModule
from embeddings.evaluator.question_answering_evaluator import QuestionAnsweringEvaluator
from embeddings.task.lightning_task.question_answering import QuestionAnsweringTask
from tests.fixtures.sample_qa_dataset import sample_question_answering_dataset

torch.manual_seed(441)


@pytest.fixture(scope="module")
def tmp_path_module(tmpdir_factory: TempdirFactory) -> Path:
    path = tmpdir_factory.mktemp(__name__)
    return Path(path)


@pytest.fixture(scope="module")
def question_answering_data_module(
    tmp_path_module: Path, sample_question_answering_dataset: datasets.DatasetDict
) -> QuestionAnsweringDataModule:
    dataset = sample_question_answering_dataset
    dataset.save_to_disk(tmp_path_module / "data_sample")
    return QuestionAnsweringDataModule(
        dataset_name_or_path=tmp_path_module / "data_sample",
        tokenizer_name_or_path="hf-internal-testing/tiny-albert",
        train_batch_size=5,
        eval_batch_size=5,
        max_seq_length=256,
        doc_stride=128,
        batch_encoding_kwargs={
            "padding": "max_length",
            "truncation": "only_second",
            "return_offsets_mapping": True,
            "return_overflowing_tokens": True,
        },
    )


@pytest.fixture(scope="module")
def question_answering_task() -> QuestionAnsweringTask:
    return QuestionAnsweringTask(
        model_name_or_path="hf-internal-testing/tiny-albert",
        finetune_last_n_layers=-1,
        task_model_kwargs={
            "optimizer": "Adam",
            "learning_rate": 1e-5,
            "adam_epsilon": 1e-8,
            "weight_decay": 0.001,
            "use_scheduler": False,
            "warmup_steps": None,
            "train_batch_size": 5,
            "eval_batch_size": 5,
        },
        output_path=".",
        model_config_kwargs={},
        task_train_kwargs={
            "max_epochs": 1,
            "devices": "auto",
            "accelerator": "auto",
            "deterministic": True,
        },
        early_stopping_kwargs={"monitor": "val/Loss", "patience": 1, "mode": "min"},
        model_checkpoint_kwargs={},
    )


@pytest.fixture(scope="module")
def scores(
    question_answering_data_module: QuestionAnsweringDataModule,
    question_answering_task: QuestionAnsweringTask,
) -> Dict[str, Any]:
    datamodule = question_answering_data_module
    task = question_answering_task
    datamodule.setup(stage="fit")
    task.build_task_model()
    task.fit(datamodule)
    datamodule.process_data(stage="test")
    predict_dataloader = datamodule.predict_dataloader()

    dataloader = predict_dataloader[0]
    model_outputs = task.predict(dataloader)
    scores = {
        "examples": datamodule.dataset_raw["validation"].to_pandas(),
        "outputs": model_outputs,
        "overflow_to_sample_mapping": datamodule.overflow_to_sample_mapping["validation"],
        "offset_mapping": datamodule.offset_mapping["validation"],
    }

    return scores


def test_question_answering_evaluator(scores: Dict[str, Any]):
    evaluator = QuestionAnsweringEvaluator()
    result = evaluator.evaluate(scores)
    validation_metrics = result.metrics
    validation_outputs = result.data
    sample_output = random.choice(validation_outputs)

    np.testing.assert_almost_equal(validation_metrics["exact"], 0.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(validation_metrics["f1"], 0.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(validation_metrics["total"], 10.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(validation_metrics["HasAns_exact"], 0.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(validation_metrics["HasAns_f1"], 0.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(validation_metrics["HasAns_total"], 9.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(validation_metrics["NoAns_exact"], 0.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(validation_metrics["NoAns_f1"], 0.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(validation_metrics["NoAns_total"], 1.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(validation_metrics["best_exact"], 10.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(
        validation_metrics["best_exact_thresh"], 0.0, decimal=pytest.decimal
    )
    np.testing.assert_almost_equal(validation_metrics["best_f1"], 10.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(
        validation_metrics["best_f1_thresh"], 0.0, decimal=pytest.decimal
    )

    assert "context" in sample_output.keys()
    assert "questions" in sample_output.keys()
    assert "answers" in sample_output.keys()
    assert "predicted_answer" in sample_output.keys()
    assert isinstance(sample_output["context"], str)
    assert isinstance(sample_output["questions"], str)
    assert sample_output["answers"] is None or isinstance(sample_output["answers"], dict)
    assert sample_output["predicted_answer"] is None or isinstance(
        sample_output["predicted_answer"], dict
    )
