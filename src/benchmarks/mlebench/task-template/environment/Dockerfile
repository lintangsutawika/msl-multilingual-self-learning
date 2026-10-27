# MLE-Bench task image ({comp_id}). Debian/Ubuntu (SAIL requirement),
# Python + mlebench + kaggle. Data is prepared at runtime in the sandbox.
FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir kaggle && pip install --no-cache-dir git+https://github.com/openai/mle-bench.git
ENV KAGGLE_USERNAME={kaggle_user}
ENV KAGGLE_KEY={kaggle_key}
WORKDIR /workspace
COPY validate_server.py /workspace/validate_server.py
COPY validate.py /workspace/validate.py
COPY validate /usr/local/bin/validate
RUN chmod +x /usr/local/bin/validate
COPY neutralize_rules.py /workspace/neutralize_rules.py

# Prepare data + start the validate server on box creation (keeps box alive).
COPY startup.sh /workspace/startup.sh
CMD ["sh", "/workspace/startup.sh"]
