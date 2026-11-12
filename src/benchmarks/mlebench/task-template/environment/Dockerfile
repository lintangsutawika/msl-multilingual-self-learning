# MLE-Bench task image ({comp_id}). Debian/Ubuntu (SAIL requirement), Python + mlebench.
# Data is BAKED into the image (environment/{public,private,leaderboard.csv} -> /workspace), or, for tasks built with
# --mount-data, the public data is bind-mounted read-only at /workspace/public and the test answers and leaderboard
# are in tests/ (uploaded only for the final grade). No runtime Kaggle download/prepare.
FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git curl sudo tmux && rm -rf /var/lib/apt/lists/*
# Harbor's Singularity bootstrap needs /usr/bin/python3, tmux and asciinema and apt-installs whatever is missing; Debian's
# python3 would then shadow this image's Python (its exec server puts /usr/bin first on PATH) and lack mlebench/pandas.
RUN ln -sf /usr/local/bin/python3 /usr/bin/python3 && ln -sf /usr/local/bin/python3 /usr/bin/python \
 && pip install --no-cache-dir asciinema
# mlebench pinned to the revision the task was built with, without its dependencies: those pull TensorFlow (GBs, slow
# image builds) that no grader uses. These are what mlebench's grading modules and the graders import.
RUN pip install --no-cache-dir --no-deps "git+https://github.com/openai/mle-bench.git@{mlebench_rev}" \
 && pip install --no-cache-dir "pandas>=2.2,<3" numpy scipy scikit-learn tqdm diskcache tenacity pyyaml appdirs py7zr \
      levenshtein pycocotools pyquaternion shapely pillow pymongo requests
WORKDIR /workspace
COPY validate_server.py /opt/msl/validate_server.py
COPY validate.py /opt/msl/validate.py
COPY bin/validate /usr/local/bin/validate
COPY bin/validate-root /usr/local/bin/validate-root
COPY score_key /etc/msl/score_key
# Babel: the host grader endpoint (validate_host.py) is bind-mounted here.
RUN mkdir -p /etc/msl-validate
COPY neutralize_rules.py /opt/msl/neutralize_rules.py
COPY grade_subset.py /opt/msl/grade_subset.py

# Start the validate server on box creation (keeps box alive).
COPY startup.sh /workspace/startup.sh

# Baked prepared data (no runtime Kaggle download).
COPY public/ /workspace/public/
COPY private/ /workspace/private/
# Validation answers (valsplit.py); holds only a placeholder when the task has no validation split.
COPY private_val/ /workspace/private_val/
COPY leaderboard.csv /workspace/leaderboard.csv

COPY validate_stub.py /workspace/validate.py

# msl-secure: the agent runs as the unprivileged user `agent` (task.toml [agent] user).
# Held-out answers, the grader and the score-signing key are root-only; `validate`
# reaches the grader through one sudo rule. /workspace is sticky: the agent can add
# files there but cannot move or replace root's.
RUN useradd -m -s /bin/bash agent \
 && printf 'agent ALL=(root) NOPASSWD: /usr/local/bin/validate-root\n' > /etc/sudoers.d/validate \
 && chmod 440 /etc/sudoers.d/validate \
 && chown -R root:root /opt/msl /etc/msl /usr/local/bin/validate-root \
 && chmod 700 /opt/msl /etc/msl /usr/local/bin/validate-root \
 && chmod 600 /etc/msl/score_key /opt/msl/validate.py \
 && chmod 755 /usr/local/bin/validate
RUN for p in /workspace/private /workspace/private_val /workspace/leaderboard.csv /workspace/validate_server.py /workspace/neutralize_rules.py /workspace/startup.sh; do \
      if [ -e "$p" ]; then chown -R root:root "$p" && chmod -R go-rwx "$p"; fi; \
    done \
 && chmod 1777 /workspace \
 && if [ -d /workspace/public ]; then chown -R agent:agent /workspace/public; fi
CMD ["sh", "/workspace/startup.sh"]
