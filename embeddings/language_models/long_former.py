from transformers import LongformerTokenizer, TFLongformerForMultipleChoice
import tensorflow as tf

tokenizer = LongformerTokenizer.from_pretrained("clarin-polish-longformer")
model = TFLongformerForMultipleChoice.from_pretrained("clarin-polish-longformer")

prompt = "Zażółć gęślą jaźń"
choice0 = "Gęś"
choice1 = "Jaźń."

encoding = tokenizer(
    [[prompt, prompt], [choice0, choice1]], return_tensors="tf", padding=True
)
inputs = {k: tf.expand_dims(v, 0) for k, v in encoding.items()}
outputs = model(inputs)  # batch size is 1

# the linear classifier still needs to be trained
logits = outputs.logits
