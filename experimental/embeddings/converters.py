from typing import List, Dict, Any


def convert_jsonl_to_connl(data: List[Dict[Any, Any]], out_path: str) -> None:
    """Convert jsonl data to connl format.

    TODO: Should be adapted for other column names"""
    for tweet in data:
        tokens = [it["text"] for it in tweet["tokens"]]
        labels = ["O" for i in range(len(tokens))]
        for span in tweet["spans"]:
            for idx in range(span["token_start"], span["token_end"] + 1):
                labels[idx] = span["label"]

        with open(out_path, "a") as f:
            for t, l in zip(tokens, labels):
                f.write(f"{t} \t  {l} \n")
            f.write("\n")
