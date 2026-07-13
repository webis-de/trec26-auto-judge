# docker build -t mam10eks/trec26-auto-judge-auto-nuggetizer:dev-0.0.1 -f judges/auto_nuggetizer/.devcontainer/Dockerfile .
FROM ubuntu:rolling

RUN apt-get update \
	&& apt-get install -y git python3.14 python3-pip 

RUN python3.14 -m pip install --break-system-packages uv 

ADD pyproject.toml /auto-judge/

WORKDIR /auto-judge

RUN uv venv --python 3.14 /venv \
	&& . /venv/bin/activate \
	&& uv pip install -e .[all]

ENV PATH=/venv/bin/:${PATH}
ENV PYTHONPATH=/auto-judge

RUN python3 -c 'import nltk; nltk.download("stopwords"); nltk.download("punkt");'

ARG NUGGETIZER_COMMIT=9a06aebdf8ddd7215e8ab7d984ea7c1eec8b1677
# The upstream package configuration currently omits nuggetizer.prompts.
RUN git clone https://github.com/mam10eks/nuggetizer.git /tmp/nuggetizer \
	&& git -C /tmp/nuggetizer checkout "${NUGGETIZER_COMMIT}" \
	&& uv pip install /tmp/nuggetizer \
	&& cp -R /tmp/nuggetizer/src/nuggetizer/prompts /venv/lib/python3.14/site-packages/nuggetizer/ \
	&& python -c 'from nuggetizer.models.nuggetizer import Nuggetizer' \
	&& rm -rf /tmp/nuggetizer

