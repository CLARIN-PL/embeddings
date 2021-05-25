# TREF - Text Representation Evaluation Framework

State-of-the-art Text Representations for Natural Language Processing tasks, an initial version of library focus on the Polish Language

# Installation

## Install poetry

### OSX / Linux / bash on Windows
```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`
```

### Windows Powershell

```bash
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
```

You only need to install Poetry once. It will automatically pick up the current Python version.

Finally, open a new shell and type the following:

```bash
poetry --version
```

If you see something like Poetry `1.1.5` then you are ready to use Poetry.

## Install python requirements

```bash
poetry install
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