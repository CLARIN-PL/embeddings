from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl

from embeddings.data.text_classification_datamodule import TextClassificationDataModule
from embeddings.model.torch_models import TransformerSimpleMLP


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
        if task_model_kwargs is None:
            task_model_kwargs = {}
        if task_train_kwargs is None:
            task_model_kwargs = {}

        self.dm = TextClassificationDataModule(
            model_name_or_path=embedding_name,
            dataset_name=dataset_name,
            text_fields=input_column_name,
            target_field=target_column_name,
            load_dataset_kwargs=load_dataset_kwargs,
        )
        self.dm.initalize()

        self.model = TransformerSimpleMLP(
            model_name_or_path=embedding_name,
            input_dim=self.dm.input_dim,
            num_labels=self.dm.num_labels,
            **self.instantiate_kwargs(task_model_kwargs),
        )
        self.trainer = pl.Trainer(**self.instantiate_kwargs(task_train_kwargs))

    @staticmethod
    def instantiate_kwargs(kwargs: Any) -> Any:
        if kwargs is None:
            kwargs = {}
        return kwargs

    def run(self) -> List[Dict[str, float]]:
        self.dm.setup("fit")
        self.trainer.fit(self.model, self.dm)
        results: List[Dict[str, float]] = self.trainer.test(datamodule=self.dm)
        return results
