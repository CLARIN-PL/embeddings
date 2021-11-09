from typing import Any, Literal

from torch import nn
from transformers import AutoConfig, AutoModel

from embeddings.embedding.document_embedding import DocumentPoolEmbedding
from embeddings.task.lightning_task.text_classification import TextClassificationTransformer


class TransformerSimpleMLP(TextClassificationTransformer):
    def __init__(
        self,
        model_name_or_path: str,
        input_dim: int,
        num_labels: int,
        pool_strategy: Literal["cls", "mean", "max"] = "cls",
        dropout_rate: float = 0.5,
        freeze_transformer: bool = True,
        **kwargs: Any
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.doc_embedder = DocumentPoolEmbedding(strategy=pool_strategy)
        if freeze_transformer:
            self.freeze_transformer()

        self.layers = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, num_labels),
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        inputs = kwargs
        outputs = self.model(**inputs)
        pooled_output = self.doc_embedder(outputs.last_hidden_state)
        # pooled_output = outputs.last_hidden_state[:, 0, :]  # take <s> token (equiv. to [CLS])
        logits = self.layers(pooled_output)
        return logits

    def unfreeze_transformer(self, unfreeze_from: int = -1) -> None:
        if unfreeze_from == -1:
            for param in self.model.base_model.parameters():
                param.requires_grad = True
        else:
            requires_grad = False
            for name, param in self.model.base_model.named_parameters():
                if not requires_grad:
                    if name.startswith("encoder.layer"):
                        no_layer = int(name.split(".")[2])
                        if no_layer >= unfreeze_from:
                            requires_grad = True
                param.requires_grad = requires_grad

    def freeze_transformer(self) -> None:
        for param in self.model.base_model.parameters():
            param.requires_grad = False
