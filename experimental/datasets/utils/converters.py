from typing import Any, Dict, List

import spacy
from spacy.training import offsets_to_biluo_tags


def convert_spacy_jsonl_to_connl_bilou(
    data: List[Dict[Any, Any]], nlp: spacy.Language, out_path: str
) -> None:
    """Convert spacy jsonl data to connl format."""
    for text in data:
        raw_text = nlp(text["text"])
        spans = text["spans"]
        entities = [(span["start"], span["end"], span["label"]) for span in spans]
        labels = offsets_to_biluo_tags(doc=raw_text, entities=entities)
        tokens = [tok.text for tok in raw_text]

        with open(out_path, "a") as f:
            for t, l in zip(tokens, labels):
                f.write(f"{t} \t {l} \n")
            f.write("\n")
