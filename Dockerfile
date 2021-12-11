FROM python:3.9.6

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
ENV PATH /root/.local/bin:$PATH

WORKDIR /code

ENV POETRY_VIRTUALENVS_CREATE false
COPY poetry.lock pyproject.toml /code/
RUN poetry install -vv --no-interaction --no-root --extras "developer pymagnitude"

ENTRYPOINT ["/bin/bash", "-c"]
