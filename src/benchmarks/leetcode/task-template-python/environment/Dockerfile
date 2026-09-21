FROM python:3.12

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3 --version