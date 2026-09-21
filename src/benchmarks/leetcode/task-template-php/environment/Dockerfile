FROM php:8.4-cli

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3 \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3 --version && \
    php --version