FROM node:22

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3 \
        ca-certificates && \
    npm install -g typescript && \
    rm -rf /var/lib/apt/lists/*

RUN python3 --version && \
    node --version && \
    npm --version && \
    tsc --version