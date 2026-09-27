FROM node:22

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3 \
        ca-certificates && \
    npm install -g @datastructures-js/priority-queue@6 @datastructures-js/queue@4.2.3 @datastructures-js/deque@1.0.4 lodash@4.17.21 && \
    rm -rf /var/lib/apt/lists/*

# Libraries LeetCode provides to JavaScript solutions.
ENV NODE_PATH=/usr/local/lib/node_modules

RUN python3 --version && \
    node --version