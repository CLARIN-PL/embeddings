from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("clarin-pl/roberta-polish-kgr10")
model = AutoModel.from_pretrained("clarin-pl/roberta-polish-kgr10")

encoded_input = tokenizer.encode("Zażółć gęślą jaźń", return_tensors="pt")
outputs = model(encoded_input)
