#!/bin/bash
# Benchmark serial vs parallel JLGridFingerprints.create on an iffSLURM node.
#
# Submit from the repo root:
#   git checkout fast-fingerprints
#   sbatch scripts/benchmark_fingerprints/benchmark_slurm.sh
#
# Rebuilds the Cython extensions on the allocated node (setup.py uses
# -march=native, so they must be compiled on the partition where the job runs).
#
# Partition choice: pick one with genuinely idle nodes, checked at submit time
# with
#   sinfo -N -h -o '%N|%P|%t|%C|%c|%m'
#   squeue -h -o '%P' | sort | uniq -c | sort -rn
# and fall back to the least-occupied queue if nothing is idle. `viti` was fully
# idle (6 nodes x 20 cores, 128 GB, empty queue) when this was first run, while
# th1-2020-64 had 59 jobs waiting and no fully-free node. Absolute wall times are
# therefore NOT comparable to reports/predictor-comparison (th1-2020-64); only
# the shape of the scaling curve is.
#
# Unlike the predictor benchmark, OMP_NUM_THREADS is deliberately NOT pinned:
# as of v0.1.5 the Cython kernels no longer link OpenMP, and leaving it unset is
# the point -- it demonstrates the footgun is gone.
#SBATCH --job-name=jl-fp-bench
#SBATCH --partition=viti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=bench_fp_%j.out
#SBATCH --error=bench_fp_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
echo "Running in $(pwd) on $(hostname); branch $(git branch --show-current)"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-} CPUS=${SLURM_CPUS_ON_NODE:-}"
grep -m1 'model name' /proc/cpuinfo || true
echo

source .venv/bin/activate
# Recompile the package's Cython extensions for this node's CPU (-march=native).
uv pip install -e . --reinstall-package jlgridfingerprints --no-deps

# Confirm the extensions really did drop OpenMP (the premise of this benchmark).
echo "--- libgomp linkage (expect 0 for all four) ---"
for so in jlgridfingerprints/lib/*.so; do
    echo -n "$(basename "$so"): "
    ldd "$so" | grep -c gomp || true
done
echo

python scripts/benchmark_fingerprints/benchmark_fingerprints.py \
    --grid 140 140 140 \
    --n-centers 2744000 \
    --num-proc 1 2 4 8 16 \
    --batch-size 20000 \
    --centers-sizes 1000 5000 13000 50000 200000 500000 \
    --centers-num-proc 8 \
    --json scripts/benchmark_fingerprints/benchmark_result.json
