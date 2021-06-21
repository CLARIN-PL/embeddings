from transformers import AutoModel, AutoTokenizer


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("clarin-pl/roberta-polish-kgr10")
    model = AutoModel.from_pretrained("clarin-pl/roberta-polish-kgr10")

    encoded_input = tokenizer.encode("Zażółć gęślą jaźń", return_tensors="pt")
    model(encoded_input)


if __name__ == "__main__":
    main()
