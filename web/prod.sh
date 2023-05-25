#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

pushd frontend
trunk build --release
popd

cargo run --bin server --release --  --addr 0.0.0.0 --port 3000 --static-dir ./dist
