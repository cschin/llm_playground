start_qudrant:
   docker run -p 6333:6333 -p 6334:6334     -e QDRANT__SERVICE__GRPC_PORT="6334"     qdrant/qdrant &
   sleep 10
   ./target/release/create_qdrant_db 

prod:
    #!/usr/bin/env bash
    # this assume the env OPENAI_API_KEY is set up
    set -euo pipefail
    IFS=$'\n\t'

    #pushd frontend
    #trunk build --release
    #popd

    pushd web/
    cargo run --bin server --release --  --addr 0.0.0.0 --port 3000 --static-dir ./dist &
    popd
    pushd python
    python demo.py
    popd

