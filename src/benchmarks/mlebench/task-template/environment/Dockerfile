# MLE-Bench task image ({comp_id}). Debian/Ubuntu (SAIL requirement), Python + mlebench.
# Data is BAKED into the image (environment/{public,private,leaderboard.csv} -> /workspace);
# no runtime Kaggle download/prepare.
FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git curl sudo && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir git+https://github.com/openai/mle-bench.git
WORKDIR /workspace
COPY validate_server.py /opt/msl/validate_server.py
COPY validate.py /opt/msl/validate.py
COPY msl-validate /usr/local/bin/validate
COPY msl-validate-root /usr/local/bin/validate-root
COPY score_key /etc/msl/score_key
COPY neutralize_rules.py /opt/msl/neutralize_rules.py

# Start the validate server on box creation (keeps box alive).
COPY startup.sh /workspace/startup.sh

# Baked prepared data (no runtime Kaggle download).
COPY public/ /workspace/public/
COPY private/ /workspace/private/
COPY leaderboard.csv /workspace/leaderboard.csv

COPY validate_stub.py /workspace/validate.py

# msl-secure: the agent runs as the unprivileged user `agent` (task.toml [agent] user).
# Held-out answers, the grader and the score-signing key are root-only; `validate`
# reaches the grader through one sudo rule. /workspace is sticky: the agent can add
# files there but cannot move or replace root's.
RUN useradd -m -s /bin/bash agent \
 && printf 'agent ALL=(root) NOPASSWD: /usr/local/bin/validate-root\n' > /etc/sudoers.d/msl-validate \
 && chmod 440 /etc/sudoers.d/msl-validate \
 && chown -R root:root /opt/msl /etc/msl /usr/local/bin/validate-root \
 && chmod 700 /opt/msl /etc/msl /usr/local/bin/validate-root \
 && chmod 600 /etc/msl/score_key /opt/msl/validate.py \
 && chmod 755 /usr/local/bin/validate
RUN for p in /workspace/private /workspace/leaderboard.csv /workspace/validate_server.py /workspace/neutralize_rules.py /workspace/startup.sh; do \
      if [ -e "$p" ]; then chown -R root:root "$p" && chmod -R go-rwx "$p"; fi; \
    done \
 && chmod 1777 /workspace \
 && if [ -d /workspace/public ]; then chown -R agent:agent /workspace/public; fi
CMD ["sh", "/workspace/startup.sh"]
