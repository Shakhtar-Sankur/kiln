"""Pools every timing per (model, engine) from bench/results/gpu_runs.jsonl
and prints the GPU median latency tables (markdown)."""

import json
import statistics
import sys
from collections import defaultdict

runs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
times = defaultdict(list)
diffs = defaultdict(float)
device = ""
for r in runs:
    key = (r["model"], r["engine"])
    times[key].extend(r["times_ms"])
    diffs[key] = max(diffs[key], r["max_diff"])
    device = device or r.get("device", "")

models = ["mlp", "bert", "llama"]
titles = {"mlp": "MLP (4 blocks, 64 x 512)", "bert": "BERT bge-small (8 x 128 tokens)",
          "llama": "SmolLM2-135M (128 tokens)"}
med = {k: statistics.median(v) for k, v in times.items()}

print(f"GPU: {device}\n")


def table(engines, title):
    engines = [e for e in engines if any((m, e) in times for m in models)]
    if not engines:
        return
    print(title + "\n")
    print("| Model | " + " | ".join(engines) + " |")
    print("|---|" + "---|" * len(engines))
    for m in models:
        have = [med[(m, e)] for e in engines if (m, e) in med]
        if not have:
            continue
        best = min(have)
        cells = []
        for e in engines:
            v = med.get((m, e))
            cells.append("-" if v is None else (f"**{v:.2f} ms**" if v == best else f"{v:.2f} ms"))
        print(f"| {titles[m]} | " + " | ".join(cells) + " |")
    print()


table(["kiln", "torch.compile", "torch.compile (CUDA graphs)", "onnxruntime", "jax/xla", "pytorch"],
      "fp32, median latency")
table(["kiln fp16", "torch.compile fp16", "pytorch fp16"],
      "fp16 matmul operands, fp32 accumulation (kiln --half; torch.autocast), median latency")
engines = [e for e in ["kiln", "kiln fp16", "torch.compile", "torch.compile (CUDA graphs)", "onnxruntime", "jax/xla",
                       "pytorch", "torch.compile fp16", "pytorch fp16"] if any((m, e) in times for m in models)]
print("Max |difference| from the PyTorch (CPU, fp32) reference:")
for m in models:
    print(f"- {m}: " + ", ".join(f"{e} {diffs[(m, e)]:.1e}" for e in engines if (m, e) in diffs))
print()
abl = [e for e in ["kiln", "kiln --no-graphs", "kiln --no-tune", "kiln --no-fusion"] if any((m, e) in med for m in models)]
print("| Model | " + " | ".join(a.replace("kiln ", "") if a != "kiln" else "kiln (all)" for a in abl) + " |")
print("|---|" + "---|" * len(abl))
for m in models:
    base = med.get((m, "kiln"))
    if base is None:
        continue
    cells = []
    for a in abl:
        v = med.get((m, a))
        cells.append("-" if v is None else (f"{v:.2f} ms" if a == "kiln" else f"{v:.2f} ms ({v / base:.2f}x)"))
    print(f"| {m} | " + " | ".join(cells) + " |")
print()
print("Timings per cell: " + ", ".join(f"{m}/{e} {len(times[(m, e)])}" for (m, e) in sorted(times)))
