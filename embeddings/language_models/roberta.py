from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("clarin-polish-roberta-tokenizer")
model = AutoModel.from_pretrained("clarin-polish-roberta-tokenizer")

encoded_input = tokenizer.encode("Zażółć gęślą jaźń", return_tensors="pt")
outputs = model(encoded_input)
