FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    tzdata \
    git \
    wget \
    vim \
    gcc \
    software-properties-common \
    cmake

RUN git clone https://github.com/CLARIN-PL/embeddings/
WORKDIR /embeddings

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils

RUN wget https://bootstrap.pypa.io/get-pip.py &&  \
	python3.9 get-pip.py && \
	ln -s /usr/bin/python3.9 /usr/local/bin/python

RUN python -m pip install poetry && \
    poetry install && \
    poetry run python -m pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

CMD exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"