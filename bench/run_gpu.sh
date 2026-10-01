#!/bin/sh
# kiln on the GPU against PyTorch, torch.compile (with and without CUDA
# graphs), ONNX Runtime (CUDA) and JAX/XLA, plus kiln ablations, on the
# three benchmark models. As in bench/run.sh, every round runs every
# configuration once per model in an order rotated from round to round,
# and the summary pools every timing.
#
# Usage: bench/run_gpu.sh [ROUNDS] [PYTHON]
set -eu
ROUNDS=${1:-3}
PY=${2:-python3}
OUT=bench/results
mkdir -p $OUT
: > $OUT/gpu_runs.jsonl
cargo build --release -q
K=./target/release/kiln
# Tune once (results are cached per GPU model), so rounds measure steady state.
for m in mlp bert llama; do
  $K run models/$m.onnx models/$m.ref --device cuda --tune --iters 1 > /dev/null
  $K run models/$m.onnx models/$m.ref --device cuda --half --tune --iters 1 > /dev/null
done
iters() { case $1 in mlp) echo 200 ;; *) echo 50 ;; esac; }
CONFIGS=${CONFIGS:-"kiln half no-attention no-graphs no-tune no-fusion baselines"}
run() {
  m=$1 c=$2 n=$(iters $1)
  case $c in
    kiln) $K run models/$m.onnx models/$m.ref --device cuda --iters $n --json $OUT/gpu_runs.jsonl --label kiln > /dev/null ;;
    half) $K run models/$m.onnx models/$m.ref --device cuda --half --iters $n --json $OUT/gpu_runs.jsonl --label "kiln fp16" > /dev/null ;;
    baselines) (cd scripts && $PY bench_gpu.py ../models $m --iters $n --json ../$OUT/gpu_runs.jsonl) ;;
    *) $K run models/$m.onnx models/$m.ref --device cuda --iters $n --$c --json $OUT/gpu_runs.jsonl --label "kiln --$c" > /dev/null ;;
  esac
}
for r in $(seq "$ROUNDS"); do
  for m in mlp bert llama; do
    set -- $CONFIGS
    k=$(( (r - 1) % $# ))
    i=0
    for c in "$@"; do
      [ $i -ge $k ] && run $m $c
      i=$((i + 1))
    done
    i=0
    for c in "$@"; do
      [ $i -lt $k ] && run $m $c
      i=$((i + 1))
    done
  done
  echo "round $r done"
done
$PY scripts/summarize_gpu.py $OUT/gpu_runs.jsonl | tee $OUT/gpu_summary.md
