#!/usr/bin/env bash
set -e

if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "Python is not installed."
    exit 1
fi

$PY_CMD src/benchmark.py --config configs/config.yaml --output results/benchmark_sample_results.json
$PY_CMD src/benchmark.py --config configs/config.yaml --sweep --output results/benchmark_sweep_results.json
