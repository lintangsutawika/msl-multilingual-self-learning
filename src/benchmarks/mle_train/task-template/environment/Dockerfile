FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends curl tmux sudo && rm -rf /var/lib/apt/lists/*
# Grading dependencies, pinned to the versions the pipeline's sanity check ran with (see requirements.txt):
# validate in the box must score exactly like the check that marked the task usable. Installed system-wide as
# root; the grader runs with `python3 -I`, so the agent's `pip install --user` packages never reach it.
RUN pip install --no-cache-dir pandas==3.1.0 numpy==2.4.6 scipy==1.17.1 scikit-learn==1.9.1 pyyaml==6.0.3 \
        pillow==13.0.0 Levenshtein==0.27.5
# Convenience libraries for the agent (not used for grading).
RUN pip install --no-cache-dir lightgbm xgboost asciinema
WORKDIR /workspace
# Grader + validation answers (public leaderboard split), the root-side validate script and the score-signing key.
COPY grader/ /opt/grader/
COPY validate.py /opt/msl/validate.py
COPY score_key /etc/msl/score_key
COPY bin/validate /usr/local/bin/validate
COPY bin/validate-root /usr/local/bin/validate-root
COPY validate_stub.py /workspace/validate.py
COPY public/ /workspace/public/

# The agent runs as the unprivileged user `agent` (task.toml [agent] user). The validation answers, the grader
# and the score-signing key are root-only; `validate` reaches the grader through one sudo rule. /workspace is
# sticky: the agent can add files there but cannot move or replace root's.
RUN useradd -m -s /bin/bash agent \
 && printf 'agent ALL=(root) NOPASSWD: /usr/local/bin/validate-root\n' > /etc/sudoers.d/validate \
 && chmod 440 /etc/sudoers.d/validate \
 && chown -R root:root /opt/grader /opt/msl /etc/msl /usr/local/bin/validate-root \
 && chmod 700 /opt/grader /opt/msl /etc/msl /usr/local/bin/validate-root \
 && chmod 600 /etc/msl/score_key /opt/msl/validate.py \
 && chmod 755 /usr/local/bin/validate \
 && chmod 644 /workspace/validate.py \
 && chmod 1777 /workspace \
 && chown -R agent:agent /workspace/public
