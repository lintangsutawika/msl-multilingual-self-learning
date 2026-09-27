FROM eclipse-temurin:21-jdk

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3 \
        libgoogle-gson-java \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3 --version && \
    javac -version && \
    test -f /usr/share/java/gson.jar