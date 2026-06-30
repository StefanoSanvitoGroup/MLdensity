#!/bin/bash
# Benchmark serial vs parallel JLPredictor on an iffSLURM compute node.
#
# Submit from the repo root:
#   git checkout fast-predictor
#   sbatch scripts/benchmark_slurm.sh
#
# Rebuilds the Cython extensions on the allocated node (setup.py uses
# -march=native, so they must be compiled on the partition where the job runs),
# pins OMP_NUM_THREADS=1 (required for num_proc>1; otherwise each worker spawns
# its own OpenMP threads and oversubscribes the cores), runs a num_proc sweep at
# the full 140^3 grid on the bundled aluminium frame, and writes JSON results.
#SBATCH --job-name=jl-bench
#SBATCH --partition=th1-2020-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=bench_%j.out
#SBATCH --error=bench_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
echo "Running in $(pwd) on $(hostname); branch $(git branch --show-current)"

source .venv/bin/activate
# Recompile the package's Cython extensions for this node's CPU (-march=native).
uv pip install -e . --reinstall-package jlgridfingerprints --no-deps

export OMP_NUM_THREADS=1
python scripts/benchmark_predictors.py \
    --grid 140 140 140 \
    --num-proc 1 2 4 8 16 32 \
    --batch-size 20000 \
    --json scripts/benchmark_result.json
