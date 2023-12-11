FROM rayproject/ray:nightly-py39-cpu

RUN sudo apt-get update && sudo apt-get install -y build-essential
RUN pip install poetry
RUN mkdir hops
WORKDIR hops

ADD pyproject.toml .
ADD poetry.lock .
RUN poetry install --no-dev --no-root

ADD . .
RUN poetry install --no-dev

# ENV SHELL /bin/bash
# ENTRYPOINT poetry run /bin/bash
