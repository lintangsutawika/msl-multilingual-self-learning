FROM python:3.12

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Pre-install what Harbor's Singularity bootstrap otherwise downloads at every
# trial start (its server venv with uvicorn/fastapi, tmux, asciinema); under load
# those per-trial downloads exceed the environment start timeout.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv tmux asciinema && \
    /usr/bin/python3 -m venv /opt/harbor-server && \
    /opt/harbor-server/bin/python3 -m pip install --no-cache-dir uvicorn fastapi && \
    rm -rf /var/lib/apt/lists/*

RUN python3 --version