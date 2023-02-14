FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /tmp

# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --no-install-recommends -y bash \
    build-essential \
    git \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists \
    && python3 -m pip install --no-cache-dir --upgrade pip poetry==1.2.2 \
    && curl -o quarto-linux-amd64.deb -L https://github.com/quarto-dev/quarto-cli/releases/download/v1.2.269/quarto-1.2.269-linux-amd64.deb \
    && apt-get update \
    && apt-get install --no-install-recommends -y gdebi-core \ 
    && gdebi -n quarto-linux-amd64.deb

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ENV PYTHONPATH=$PYTHONPATH:/app

WORKDIR /app

COPY poetry.lock .
COPY pyproject.toml .

RUN conda create -n embeddings python=3.9 \
    && echo "source activate embeddings" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

RUN pip install --no-cache-dir poetry==1.2.2 \
    && poetry install --no-interaction \
    && poetry run poe force-torch-cuda

CMD ["poetry", "run", "python", "-m", "jupyter", "notebook", "--ip='*'", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]
