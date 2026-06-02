#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ENTRYPOINT="${ENTRYPOINT:-train}"

case "$ENTRYPOINT" in
  train)
    cmd=("$PYTHON_BIN" train.py --help)
    ;;
  evaluate)
    cmd=("$PYTHON_BIN" evaluate.py --help)
    ;;
  benchmark)
    cmd=("$PYTHON_BIN" evaluate_benchmark.py --help)
    ;;
  grid_search)
    cmd=("$PYTHON_BIN" grid_search_benchmark.py --help)
    ;;
  pf_generate)
    cmd=("$PYTHON_BIN" gen_pf_results.py --help)
    ;;
  pf_process)
    cmd=("$PYTHON_BIN" psef/plotting/pf_results.py --help)
    ;;
  *)
    echo "Unknown ENTRYPOINT='$ENTRYPOINT'. Use train, evaluate, benchmark, grid_search, pf_generate, or pf_process." >&2
    exit 2
    ;;
esac

"${cmd[@]}"
