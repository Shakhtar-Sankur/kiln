"""Times the same models with ONNX Runtime, PyTorch (eager and
torch.compile) and JAX/XLA, on the same inputs, with the same thread count,
and checks each against the reference output.

Usage: python scripts/bench_baselines.py MODELS_DIR NAME [--threads 4] [--iters 20]
       [--engines ort,torch,compile,jax] [--json OUT]
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def read_ref(path):
    b = open(path, "rb").read()
    assert b[:8] == b"KILNREF1"
    hl = struct.unpack("<I", b[8:12])[0]
    meta = json.loads(b[12:12 + hl])
    pos = 12 + hl
    out = {}
    for kind in ("inputs", "outputs"):
        out[kind] = []
        for t in meta[kind]:
            dt = np.float32 if t["dtype"] == "f32" else np.int64
            n = int(np.prod(t["shape"])) if t["shape"] else 1
            a = np.frombuffer(b, dtype=dt, count=n, offset=pos).reshape(t["shape"])
            pos += a.nbytes
            out[kind].append((t["name"], a.copy()))
    return out


def timeit(f, iters):
    f()
    f()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    s = sorted(ts)
    return s[len(s) // 2], s[0], ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models")
    ap.add_argument("name")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--engines", default="ort,torch,compile,jax")
    ap.add_argument("--json")
    a = ap.parse_args()
    ref = read_ref(os.path.join(a.models, a.name + ".ref"))
    feeds = dict(ref["inputs"])
    want = ref["outputs"][0][1]
    results = []

    def report(engine, out, med, best, ts):
        err = float(np.abs(np.asarray(out, dtype=np.float32) - want).max())
        print(f"{a.name} {engine}: median {med * 1e3:.2f} ms, min {best * 1e3:.2f} ms, max |diff| vs reference {err:.2e}")
        results.append({"model": a.name, "engine": engine, "median_ms": med * 1e3, "min_ms": best * 1e3,
                        "max_diff": err, "threads": a.threads, "times_ms": [round(t * 1e3, 3) for t in ts]})

    engines = a.engines.split(",")
    if "ort" in engines:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = a.threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        s = ort.InferenceSession(os.path.join(a.models, a.name + ".onnx"), so, providers=["CPUExecutionProvider"])
        run = lambda: s.run(None, feeds)[0]
        out = run()
        report("onnxruntime", out, *timeit(run, a.iters))
    if "torch" in engines or "compile" in engines or "jax" in engines:
        import torch
        torch.set_num_threads(a.threads)
        import baseline_models as bm
        model, args = bm.torch_model(a.name, feeds)
        with torch.no_grad():
            if "torch" in engines:
                run = lambda: model(*args)
                out = run().numpy()
                report("pytorch", out, *timeit(run, a.iters))
            if "compile" in engines:
                cm = torch.compile(model)
                run = lambda: cm(*args)
                out = run().numpy()
                report("torch.compile", out, *timeit(run, a.iters))
        if "jax" in engines:
            os.environ.setdefault("XLA_FLAGS", f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={a.threads}")
            fn, jargs = bm.jax_model(a.name, model, feeds)
            run = lambda: fn(*jargs).block_until_ready()
            out = np.asarray(run())
            report("jax/xla", out, *timeit(run, a.iters))
    if a.json:
        with open(a.json, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
