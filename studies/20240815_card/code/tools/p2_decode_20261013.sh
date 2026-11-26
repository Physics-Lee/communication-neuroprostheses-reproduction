#!/bin/bash
# Run the indexed WFST stage from a WSL Python environment.
set -eu
tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python3}" "$tools/decode_card_run.py" p2 "$@"
