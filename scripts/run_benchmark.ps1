$ErrorActionPreference = "Stop"

$pyCmd = "python"
if (-not (Get-Command $pyCmd -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not found in PATH."
    exit 1
}

& $pyCmd src/benchmark.py --config configs/config.yaml --output results/benchmark_sample_results.json
& $pyCmd src/benchmark.py --config configs/config.yaml --sweep --output results/benchmark_sweep_results.json
