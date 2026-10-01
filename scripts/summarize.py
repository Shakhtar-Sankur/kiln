"""Pools every timing per (model, engine) from bench/results/runs.jsonl and
prints the median latency tables (markdown)."""

import json
import statistics
import sys
from collections import defaultdict

runs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
times = defaultdict(list)
diffs = defaultdict(float)
extra = {}
for r in runs:
    key = (r["model"], r["engine"])
    times[key].extend(r["times_ms"])
    diffs[key] = max(diffs[key], r["max_diff"])
    if r["engine"] == "kiln":
        extra[r["model"]] = r

models = ["mlp", "bert", "llama"]
titles = {"mlp": "MLP (4 blocks, 64 x 512)", "bert": "BERT bge-small (8 x 128 tokens)", "llama": "SmolLM2-135M (128 tokens)"}
engines = ["kiln", "onnxruntime", "pytorch", "torch.compile", "jax/xla"]
med = {k: statistics.median(v) for k, v in times.items()}

print("| Model | " + " | ".join(engines) + " |")
print("|---|" + "---|" * len(engines))
for m in models:
    cells = []
    best = min(med[(m, e)] for e in engines if (m, e) in med)
    for e in engines:
        if (m, e) not in med:
            cells.append("-")
            continue
        v = med[(m, e)]
        cells.append(f"**{v:.1f} ms**" if v == best else f"{v:.1f} ms")
    print(f"| {titles[m]} | " + " | ".join(cells) + " |")
print()
print("Max |difference| from the PyTorch reference:")
for m in models:
    print(f"- {m}: " + ", ".join(f"{e} {diffs[(m, e)]:.1e}" for e in engines if (m, e) in diffs))
print()
abl = ["kiln", "kiln --no-fusion", "kiln --no-epilogue", "kiln --no-rows", "kiln --no-inline", "kiln --no-fold", "kiln --no-tune"]
print("| Model | " + " | ".join(a.replace("kiln ", "") if a != "kiln" else "kiln (all)" for a in abl) + " |")
print("|---|" + "---|" * len(abl))
for m in models:
    base = med.get((m, "kiln"))
    cells = []
    for a in abl:
        v = med.get((m, a))
        if v is None:
            cells.append("-")
        elif a == "kiln":
            cells.append(f"{v:.1f} ms")
        else:
            cells.append(f"{v:.1f} ms ({v / base:.2f}x)")
    print(f"| {m} | " + " | ".join(cells) + " |")
print()
for m in models:
    if m in extra:
        x = extra[m]
        print(f"- {m}: {x['kernels']} kernels; arena {x['arena_mb']:.1f} MB for {x['intermediate_mb']:.1f} MB of intermediates")
print()
print("Samples per engine: " + ", ".join(f"{k[0]}/{k[1]} {len(v)}" for k, v in sorted(times.items())))
