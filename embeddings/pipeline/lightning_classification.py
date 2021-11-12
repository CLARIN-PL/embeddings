from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch

from embeddings.data.text_classification_datamodule import TextClassificationDataModule
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.model.torch_models import AutoTransformerForSequenceClassification


class TorchClassificationPipeline:
    def __init__(
        self,
        embedding_name: str,
        dataset_name: str,
        input_column_name: Union[str, List[str]],
        target_column_name: str,
        # output_path: T_path,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.dm = TextClassificationDataModule(
            model_name_or_path=embedding_name,
            dataset_name=dataset_name,
            text_fields=input_column_name,
            target_field=target_column_name,
            load_dataset_kwargs=load_dataset_kwargs,
        )
        self.dm.initalize()

        self.model = AutoTransformerForSequenceClassification(
            model_name_or_path=embedding_name,
            input_dim=self.dm.input_dim,
            num_labels=self.dm.num_labels,
            **self.instantiate_kwargs(task_model_kwargs),
        )
        self.trainer = pl.Trainer(**self.instantiate_kwargs(task_train_kwargs))
        self.evaluator = TextClassificationEvaluator()

    @staticmethod
    def instantiate_kwargs(kwargs: Any) -> Any:
        if kwargs is None:
            kwargs = {}
        return kwargs

    def predict(self, **kwargs: Any) -> Dict[str, np.ndarray]:
        predictions = self.trainer.predict(**kwargs)
        predictions = torch.argmax(torch.cat(predictions), dim=1).numpy()
        ground_truth = torch.cat([x["labels"] for x in list(self.dm.test_dataloader())]).numpy()
        return {"y_pred": predictions, "y_true": ground_truth}

    def run(self) -> Dict[str, Any]:
        self.dm.setup("fit")
        self.trainer.fit(self.model, self.dm)
        self.trainer.test(datamodule=self.dm)
        model_result = self.predict(dataloaders=self.dm.test_dataloader())
        metrics = self.evaluator.evaluate(model_result)
        return metrics
