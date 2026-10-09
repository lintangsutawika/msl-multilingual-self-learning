FROM node:22

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3 \
        ca-certificates && \
    npm install -g typescript@5 @datastructures-js/priority-queue@6 @datastructures-js/queue@4.2.3 @datastructures-js/deque@1.0.4 lodash@4.17.21 && \
    rm -rf /var/lib/apt/lists/*

# Libraries LeetCode provides to TypeScript solutions.
ENV NODE_PATH=/usr/local/lib/node_modules

# Pre-install what Harbor's Singularity bootstrap otherwise downloads at every
# trial start (its server venv with uvicorn/fastapi, tmux, asciinema); under load
# those per-trial downloads exceed the environment start timeout.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv tmux asciinema && \
    /usr/bin/python3 -m venv /opt/harbor-server && \
    /opt/harbor-server/bin/python3 -m pip install --no-cache-dir uvicorn fastapi && \
    rm -rf /var/lib/apt/lists/*

RUN python3 --version && \
    node --version && \
    npm --version && \
    tsc --version