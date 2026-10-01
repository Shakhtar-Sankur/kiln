#!/bin/sh
# kiln against ONNX Runtime, PyTorch, torch.compile and JAX/XLA, plus kiln
# ablations, on the three benchmark models. Every round runs every
# configuration once per model, in an order rotated from round to round,
# so that neither slow periods of the machine nor position in the sequence
# favour any engine; the summary pools every timing. Needs models/ from
# scripts/export_models.py and Python with onnxruntime, torch,
# transformers and jax.
#
# Usage: bench/run.sh [ROUNDS] [PYTHON]
set -eu
ROUNDS=${1:-3}
PY=${2:-python3}
OUT=bench/results
mkdir -p $OUT
: > $OUT/runs.jsonl
cargo build --release -q
K=./target/release/kiln
# Tune once (results are cached), so rounds measure steady state.
for m in mlp bert llama; do
  $K run models/$m.onnx models/$m.ref --tune --iters 1 > /dev/null
done
iters() { case $1 in mlp) echo 100 ;; *) echo 20 ;; esac; }
CONFIGS="kiln no-tune no-epilogue no-rows no-inline no-fold no-fusion baselines"
run() {
  m=$1 c=$2 n=$(iters $1)
  case $c in
    kiln) $K run models/$m.onnx models/$m.ref --iters $n --json $OUT/runs.jsonl --label kiln > /dev/null ;;
    baselines) (cd scripts && $PY bench_baselines.py ../models $m --iters $n --json ../$OUT/runs.jsonl > /dev/null 2>&1) ;;
    *) $K run models/$m.onnx models/$m.ref --iters $n --$c --json $OUT/runs.jsonl --label "kiln --$c" > /dev/null ;;
  esac
}
for r in $(seq "$ROUNDS"); do
  for m in mlp bert llama; do
    # This round's order: the configurations rotated by the round number.
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
$PY scripts/summarize.py $OUT/runs.jsonl | tee $OUT/summary.md
