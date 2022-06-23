from pathlib import Path

from onnxruntime import InferenceSession
from transformers import AutoTokenizer
import numpy as np
from embeddings.model.lightning_module.text_classification import TextClassificationModule
from embeddings.utils.onnx_exporter import OnnxExportConfiguration, OnnxExporter


def test_export() -> None:
    exporter = OnnxExporter()
    model = "allegro/herbert-base-cased"
    fake_task_kwargs = {
        "optimizer": "adam",
        "learning_rate": 0,
        "adam_epsilon": 1,
        "warmup_steps": 1,
        "weight_decay": 1,
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "use_scheduler": False,
    }
    module = TextClassificationModule(
        model_name_or_path=model,
        num_classes=2,
        finetune_last_n_layers=-1,
        task_model_kwargs=fake_task_kwargs,
    )
    config = OnnxExportConfiguration(Path("/tmp/"), model, module)
    exporter.export(config)

    tokenizer = AutoTokenizer.from_pretrained(model)
    session = InferenceSession("/tmp/model.onnx")
    inputs = tokenizer(f"Using {model} with ONNX Runtime!", return_tensors="np")
    outputs = session.run(None, input_feed=dict(inputs))
    assert len(outputs) == 1
