FROM gcc:13

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3 \
        nlohmann-json3-dev \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# toolchain smoke test (%test): fails the build if any is missing
RUN python3 --version && \
    g++ --version && \
    test -f /usr/include/nlohmann/json.hpp