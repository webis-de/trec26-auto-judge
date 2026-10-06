# The base image is build from .devcontainer/Dockerfile
FROM ghcr.io/trec-auto-judge/trec-auto-judge-base:dev-0.0.1

ADD judges /auto-judge/judges
ADD pyproject.toml /auto-judge/

export TIMEOUT_S=900
export MAX_OUTSTANDING=16
export BATCH_MAX_FAILURES=200
WORKDIR /auto-judge

RUN uv pip install --system -e .[all]

