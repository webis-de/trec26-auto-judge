FROM mam10eks/trec26-auto-judge-auto-nuggetizer:dev-0.0.1

ADD judges /auto-judge/judges

WORKDIR /auto-judge

RUN python3 -m pytest -q judges/auto_nuggetizer/test_auto_nuggetizer.py

