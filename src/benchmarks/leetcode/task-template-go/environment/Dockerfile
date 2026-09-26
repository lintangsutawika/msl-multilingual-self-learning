FROM golang:1.24

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3 \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3 --version && \
    go version