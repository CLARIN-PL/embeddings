FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04
MAINTAINER Lukasz Augustyniak <lukasz.augustyniak@pwr.edu.pl>

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    jupyter \
    tensorflow \
    torch \
    transformers
