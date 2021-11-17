from typing import Any, Literal, Optional

import torch
from torch import FloatTensor, nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput

from embeddings.embedding.document_embedding import DocumentPoolEmbedding
from embeddings.model.lightning.core import Transformer
from embeddings.model.lightning.sequence_classification import SequenceClassificationModule


class MLPTransformerForSequenceClassification(Transformer, SequenceClassificationModule):
    def __init__(
        self,
        model_name_or_path: str,
        input_dim: int,
        num_labels: int,
        pool_strategy: Literal["cls", "mean", "max"] = "cls",
        dropout_rate: float = 0.5,
        unfreeze_from: Optional[int] = None,
        **kwargs: Any
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.doc_embedder = DocumentPoolEmbedding(strategy=pool_strategy)
        self.layers = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, num_labels),
        )
        self.freeze_transformer()
        if unfreeze_from:
            self.unfreeze_transformer(unfreeze_from=unfreeze_from)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        *args,
        **kwargs
    ) -> Any:
        # assert not (args and kwargs)
        # assert args or kwargs
        # inputs = kwargs if kwargs else args
        # if isinstance(inputs, tuple):
        #     inputs = dict(ChainMap(*inputs))
        # labels = inputs.pop("labels", None)

        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = self.doc_embedder(outputs.last_hidden_state)
        # pooled_output = outputs.last_hidden_state[:, 0, :]  # take <s> token (equiv. to [CLS])
        logits = self.layers(pooled_output)

        loss = self.calculate_loss(logits=logits, labels=labels, attention_mask=attention_mask)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def calculate_loss(self, logits, labels, attention_mask) -> Optional[FloatTensor]:
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss
