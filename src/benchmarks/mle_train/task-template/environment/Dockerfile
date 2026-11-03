FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends curl tmux && rm -rf /var/lib/apt/lists/*
# Grading dependencies, pinned to the versions the pipeline's sanity check ran with (see requirements.txt):
# validate in the box must score exactly like the check that marked the task usable.
RUN pip install --no-cache-dir pandas==3.1.0 numpy==2.4.6 scipy==1.17.1 scikit-learn==1.9.1 pyyaml==6.0.3 \
        pillow==13.0.0 Levenshtein==0.27.5
# Convenience libraries for the agent (not used for grading).
RUN pip install --no-cache-dir lightgbm xgboost asciinema
# Harbor's Singularity bootstrap puts /usr/bin first on PATH; make /usr/bin/python3 the image's Python.
RUN ln -sf /usr/local/bin/python3 /usr/bin/python3
WORKDIR /workspace
COPY grader/ /opt/grader/
COPY validate /usr/local/bin/validate
RUN chmod +x /usr/local/bin/validate
COPY public/ /workspace/public/
