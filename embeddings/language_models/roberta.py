from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("clarin/roberta-polish-v1")
model = AutoModel.from_pretrained("clarin/roberta-polish-v1")

encoded_input = tokenizer.encode("Zażółć gęślą jaźń", return_tensors="pt")
outputs = model(encoded_input)
