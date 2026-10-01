"""Draws the README's charts (SVG, no dependencies) from
bench/results/runs.jsonl. Usage: python3 scripts/charts.py"""

import json
import statistics
from collections import defaultdict

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
STRONG = "#2a78d6"
WEAK = "#9ec5f4"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
W = 720


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def grouped(path, title, subtitle, groups, unit, note):
    """Horizontal bars in sections: groups is [(section title, [(label, value, strong)])];
    each section has its own scale."""
    left, right, band, head = 150, 120, 30, 34
    h = 80 + sum(head + band * len(rows) + 14 for _, rows in groups) + 30
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}" font-family="{FONT}" role="img" aria-label="{esc(title)}">',
        f'<rect width="{W}" height="{h}" rx="8" fill="{SURFACE}"/>',
        f'<text x="24" y="34" font-size="17" font-weight="600" fill="{TEXT}">{esc(title)}</text>',
        f'<text x="24" y="56" font-size="13" fill="{TEXT2}">{esc(subtitle)}</text>',
    ]
    y0 = 80
    plot = W - left - right
    for gt, rows in groups:
        out.append(f'<text x="24" y="{y0 + 18}" font-size="13" font-weight="600" fill="{TEXT}">{esc(gt)}</text>')
        y0 += head
        vmax = max(v for _, v, _ in rows) * 1.05
        for i, (label, v, strong) in enumerate(rows):
            y = y0 + band * i + (band - 20) / 2
            w = max(plot * v / vmax, 4)
            color = STRONG if strong else WEAK
            out.append(
                f'<path d="M{left},{y} h{w - 4:.1f} a4,4 0 0 1 4,4 v12 a4,4 0 0 1 -4,4 h{-(w - 4):.1f} z" fill="{color}">'
                f"<title>{esc(label)}: {v:.1f} {esc(unit)}</title></path>"
            )
            out.append(f'<text x="{left - 12}" y="{y + 14}" font-size="13" fill="{TEXT}" text-anchor="end">{esc(label)}</text>')
            out.append(f'<text x="{left + w + 8:.1f}" y="{y + 14}" font-size="13" font-weight="600" fill="{TEXT}">{v:.1f} <tspan font-weight="400" fill="{TEXT2}">{esc(unit)}</tspan></text>')
        y0 += band * len(rows) + 14
    out.append(f'<text x="24" y="{h - 14}" font-size="12" fill="{TEXT2}">{esc(note)}</text>')
    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out))


def main():
    runs = [json.loads(l) for l in open("bench/results/runs.jsonl") if l.strip()]
    times = defaultdict(list)
    for r in runs:
        times[(r["model"], r["engine"])].extend(r["times_ms"])
    med = {k: statistics.median(v) for k, v in times.items()}
    titles = {"bert": "BERT, bge-small-en-v1.5: 8 sequences x 128 tokens",
              "llama": "SmolLM2-135M: 128-token prefill",
              "mlp": "MLP: 4 pre-norm GELU blocks, 64 x 512"}
    engines = [("kiln", "kiln"), ("torch.compile", "torch.compile"), ("onnxruntime", "ONNX Runtime"),
               ("jax/xla", "JAX / XLA"), ("pytorch", "PyTorch eager")]
    groups = []
    for m in ("bert", "llama", "mlp"):
        rows = [(label, med[(m, e)], e == "kiln") for e, label in engines if (m, e) in med]
        groups.append((titles[m], rows))
    grouped("docs/latency.svg", "Inference latency: kiln against four production stacks",
            "Median milliseconds, lower is better; same machine, same 4 threads, same inputs", groups, "ms",
            "4 vCPUs (Xeon, AVX-512). Every engine's output is checked against PyTorch's before timing.")
    abl = [("kiln", "everything"), ("kiln --no-fusion", "no fusion at all"), ("kiln --no-epilogue", "no epilogue fusion"),
           ("kiln --no-rows", "no row fusion"), ("kiln --no-fold", "no batch folding"), ("kiln --no-tune", "no tuning")]
    groups = []
    for m in ("bert", "llama", "mlp"):
        rows = [(label, med[(m, e)], e == "kiln") for e, label in abl if (m, e) in med]
        if rows:
            groups.append((titles[m], rows))
    grouped("docs/ablations.svg", "What each optimization is worth",
            "Median milliseconds with one optimization switched off, lower is better", groups, "ms",
            "Each bar switches off one pass (fusion: all three of epilogue, row and inline fusion).")


if __name__ == "__main__":
    main()
