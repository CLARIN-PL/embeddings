State-of-the-art Text Representations for Natural Language Processing tasks, an initial version of library focus on the Polish Language

# Installation

```bash
pip install clarinpl-embeddings
```

# Run example tasks

```bash
cd .\examples
```

## Run classification task


```python
python evaluate_document_classification.py --embedding-name allegro/herbert-base-cased --dataset-name clarin-pl/polemo2-official --input-column-name text --target-column-name target
```

## Run sequence tagging task

to be announced