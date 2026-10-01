#!/bin/bash
# kiln on a Google Colab GPU (or any Linux machine with an NVIDIA GPU, CUDA
# and Python with torch and transformers): builds kiln, runs the GPU test
# suite on the GPU, tunes, profiles and benchmarks against PyTorch,
# torch.compile, ONNX Runtime and JAX/XLA, and writes everything to
# colab_report.txt (printed at the end).
#
# In Colab (Runtime > Change runtime type > T4 GPU), one cell:
#   !git clone https://github.com/Shakhtar-Sankur/kiln && cd kiln && bash scripts/colab.sh
#
# Usage: bash scripts/colab.sh [ROUNDS]
set -u
ROUNDS=${1:-3}
cd "$(dirname "$0")/.."
REPORT=$PWD/colab_report.txt
: > "$REPORT"
log() { echo "$@" | tee -a "$REPORT"; }
step() { log; log "=== $* ==="; }

step "machine"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv 2>&1 | tee -a "$REPORT"
nproc | tee -a "$REPORT"
git log -1 --format='kiln %h %s' | tee -a "$REPORT"

step "toolchain"
if ! command -v cargo > /dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > /dev/null 2>&1
fi
source "$HOME/.cargo/env"
rustc --version | tee -a "$REPORT"
pip install -q onnx onnxruntime-gpu 2>&1 | tail -2
python3 -c "import torch, transformers, onnxruntime as o; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__, '| onnxruntime', o.__version__, o.get_available_providers())" 2>&1 | tee -a "$REPORT"
python3 -c "import jax; print('jax', jax.__version__, jax.devices())" 2>&1 | tail -1 | tee -a "$REPORT"
ls /usr/local/cuda/lib64/libnvrtc.so* 2>/dev/null | head -1 | tee -a "$REPORT"

step "models"
export KILN_MODELS=$PWD/models
python3 - <<'PY' 2>&1 | tail -2
from huggingface_hub import snapshot_download
snapshot_download("BAAI/bge-small-en-v1.5", local_dir="models/bge-small")
snapshot_download("HuggingFaceTB/SmolLM2-135M", local_dir="models/smollm2-135m")
PY
python3 scripts/export_models.py models mlp bert llama 2>&1 | grep -v Warning | tail -5 | tee -a "$REPORT"

step "build"
cargo build --release 2>&1 | tail -2 | tee -a "$REPORT"
K=./target/release/kiln

step "GPU tests (emulator and this GPU)"
cargo test --release --test gpu -- --nocapture 2>&1 | grep -E "^test |test result|panicked|also testing|no GPU|max \|diff\||error" | tee -a "$REPORT"

step "first run, tuning and per-kernel profile"
for m in mlp bert llama; do
  log "--- $m"
  $K run models/$m.onnx models/$m.ref --device cuda --tune --iters 20 --profile 2>&1 | tee -a "$REPORT"
done
for m in mlp bert llama; do
  log "--- $m untuned defaults, no CUDA graphs"
  $K run models/$m.onnx models/$m.ref --device cuda --no-tune --no-graphs --iters 20 2>&1 | grep -E "diff|latency" | tee -a "$REPORT"
done
log "--- CPU backend on this machine, for reference"
for m in mlp bert llama; do
  $K run models/$m.onnx models/$m.ref --iters 5 2>&1 | grep -E "latency" | tee -a "$REPORT"
done

step "benchmark ($ROUNDS rounds)"
bench/run_gpu.sh "$ROUNDS" python3 2>&1 | grep -v "^round" | tee -a "$REPORT"

step "raw timings (bench/results/gpu_runs.jsonl)"
cat bench/results/gpu_runs.jsonl >> "$REPORT"
cat "$HOME/.cache/kiln/gtune.txt" >> "$REPORT" 2>/dev/null

echo
echo "=================== KILN COLAB REPORT BEGIN ==================="
cat "$REPORT"
echo "=================== KILN COLAB REPORT END ====================="
