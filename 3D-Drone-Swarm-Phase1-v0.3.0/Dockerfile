FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace
COPY . /workspace
RUN python -m pip install --upgrade pip \
    && python -m pip install ".[pipeline]"

RUN useradd --create-home --uid 10001 swarm \
    && chown -R swarm:swarm /workspace
USER swarm

ENTRYPOINT ["drone-swarm"]
CMD ["--help"]
