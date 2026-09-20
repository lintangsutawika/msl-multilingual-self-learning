#!/usr/bin/env bash
set -euo pipefail

echo '  -- python3 --version'
python3 --version

echo '  -- rustc --version'
rustc --version

echo '  -- cargo --version'
cargo --version

test -d /usr/share/cargo/registry

mkdir -p /workspace/.cargo

cat > /workspace/.cargo/config.toml <<'CARGO'
[source.crates-io]
replace-with = "debian"

[source.debian]
directory = "/usr/share/cargo/registry"
CARGO

mkdir -p /opt/leetcode
cp /staging/env_files/adapters/* /opt/leetcode/

echo "[setup] Rust environment ready"
