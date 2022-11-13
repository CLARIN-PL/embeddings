FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --no-install-recommends -y bash \
    build-essential \
    git \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists \
    && python3 -m pip install --no-cache-dir --upgrade pip poetry==1.2.2

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV \
    && $VIRTUAL_ENV/bin/python -m pip install --upgrade pip \
    && poetry config virtualenvs.path $VIRTUAL_ENV 

ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH=$PYTHONPATH:/app

WORKDIR /tmp
RUN curl -o quarto-linux-amd64.deb -L https://github.com/quarto-dev/quarto-cli/releases/download/v1.2.269/quarto-1.2.269-linux-amd64.deb \
    && apt-get update \
    && apt-get install --no-install-recommends -y gdebi-core \ 
    && gdebi -n quarto-linux-amd64.deb

WORKDIR /app

COPY poetry.lock .
COPY pyproject.toml .

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi


CMD ["poetry", "run", "python", "-m", "jupyter", "notebook", "--ip='*'", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]
