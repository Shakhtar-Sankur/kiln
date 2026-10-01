"""Draws the README's GPU charts (SVG, no dependencies) from
bench/results/gpu_runs.jsonl. Usage: python3 scripts/charts_gpu.py [RUNS]"""

import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from charts import grouped  # noqa: E402

TITLES = {"bert": "BERT, bge-small-en-v1.5: 8 sequences x 128 tokens",
          "llama": "SmolLM2-135M: 128-token prefill",
          "mlp": "MLP: 4 pre-norm GELU blocks, 64 x 512"}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "bench/results/gpu_runs.jsonl"
    runs = [json.loads(l) for l in open(path) if l.strip()]
    times = defaultdict(list)
    device = ""
    for r in runs:
        times[(r["model"], r["engine"])].extend(r["times_ms"])
        device = device or r.get("device", "")
    med = {k: statistics.median(v) for k, v in times.items()}
    gpu = device.split(" (")[0] or "GPU"

    def section(engines):
        groups = []
        for m in ["bert", "llama", "mlp"]:
            rows = [(label, med[(m, e)], e.startswith("kiln")) for e, label in engines if (m, e) in med]
            if rows:
                groups.append((TITLES[m], rows))
        return groups

    fp16 = [("kiln fp16", "kiln"), ("torch.compile fp16", "torch.compile"), ("pytorch fp16", "PyTorch eager")]
    fp32 = [("kiln", "kiln"), ("torch.compile (CUDA graphs)", "torch.compile"), ("onnxruntime", "ONNX Runtime"),
            ("jax/xla", "JAX / XLA"), ("pytorch", "PyTorch eager")]
    grouped("docs/gpu_fp16.svg", f"{gpu}, fp16 tensor cores: kiln against torch.compile and PyTorch",
            "Median milliseconds, lower is better; fp16 matmul operands, fp32 accumulation (torch.autocast)",
            section(fp16), "ms", "Every engine's output is checked against PyTorch's before timing; bench/run_gpu.sh.",
            digits=2)
    grouped("docs/gpu_fp32.svg", f"{gpu}, fp32: kiln against four production stacks",
            "Median milliseconds, lower is better; same GPU, same inputs, outputs left on the device",
            section(fp32), "ms", "torch.compile shown with CUDA graphs (mode=reduce-overhead); bench/run_gpu.sh.",
            digits=2)


if __name__ == "__main__":
    main()
