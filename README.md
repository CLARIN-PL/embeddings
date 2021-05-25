# Installation

## Install poetry

### osx / linux / bash on windows
```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`
```

### windows powershell install instructions

```bash
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
```

You only need to install Poetry once. It will automatically pick up the current Python version.

Finally, open a new shell and type the following:

```bash
poetry --version`
```

If you see something like Poetry `1.1.5` then you are ready to use Poetry.

## Install python requirements

```bash
poetry install
```

# Run examples tasks

## Run classification task

## Run sequence tagging task