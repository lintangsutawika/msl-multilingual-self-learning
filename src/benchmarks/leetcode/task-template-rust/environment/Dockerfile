FROM rust:1.90

ENV PATH=/usr/bin:$PATH

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3 \
        librust-serde-dev \
        librust-serde-json-dev \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3 --version && \
    rustc --version && \
    cargo --version && \
    test -d /usr/share/cargo/registry