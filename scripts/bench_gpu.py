"""Times the benchmark models on the GPU with PyTorch (eager, torch.compile
and torch.compile with CUDA graphs), ONNX Runtime's CUDA provider and
JAX/XLA, on the same inputs, checking each against the reference output.

Every engine runs with its inputs already on the device and leaves its
outputs there; a timing is one call plus waiting for the device, as for
`kiln run --device cuda`.

Usage: python scripts/bench_gpu.py MODELS_DIR NAME [--iters 50]
       [--engines torch,compile,compile-graphs,ort,jax,torch-fp16,compile-fp16] [--json OUT]

The -fp16 engines run under torch.autocast(float16): matmuls with fp16
operands and fp32 accumulation, as `kiln run --half`.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bench_baselines import read_ref  # noqa: E402


def timeit(f, sync, iters):
    for _ in range(3):
        f()
    sync()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        f()
        sync()
        ts.append(time.perf_counter() - t)
    s = sorted(ts)
    return s[len(s) // 2], s[0], ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models")
    ap.add_argument("name")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--engines", default="torch,compile,compile-graphs,ort,jax,torch-fp16,compile-fp16")
    ap.add_argument("--json")
    a = ap.parse_args()
    ref = read_ref(os.path.join(a.models, a.name + ".ref"))
    feeds = dict(ref["inputs"])
    want = ref["outputs"][0][1]
    results = []
    engines = a.engines.split(",")

    def report(engine, out, med, best, ts, device):
        err = float(np.abs(np.asarray(out, dtype=np.float32) - want).max())
        print(f"{a.name} {engine}: median {med * 1e3:.3f} ms, min {best * 1e3:.3f} ms, max |diff| {err:.2e}")
        results.append({"model": a.name, "engine": engine, "device": device, "median_ms": med * 1e3,
                        "min_ms": best * 1e3, "max_diff": err, "times_ms": [round(t * 1e3, 4) for t in ts]})

    import torch
    dev_name = torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if {"torch", "compile", "compile-graphs", "jax", "torch-fp16", "compile-fp16"} & set(engines):
        import baseline_models as bm
        model, args = bm.torch_model(a.name, feeds)
        model = model.cuda()
        args = tuple(x.cuda() for x in args)
        sync = torch.cuda.synchronize
        with torch.no_grad():
            if "torch" in engines:
                run = lambda: model(*args)
                report("pytorch", run().cpu().numpy(), *timeit(run, sync, a.iters), dev_name)
            if "compile" in engines:
                cm = torch.compile(model)
                run = lambda: cm(*args)
                report("torch.compile", run().cpu().numpy(), *timeit(run, sync, a.iters), dev_name)
            if "compile-graphs" in engines:
                torch._dynamo.reset()
                cm = torch.compile(model, mode="reduce-overhead")
                run = lambda: cm(*args)
                run()
                report("torch.compile (CUDA graphs)", run().cpu().numpy(), *timeit(run, sync, a.iters), dev_name)
            for e, mode in (("torch-fp16", None), ("compile-fp16", "default")):
                if e not in engines:
                    continue
                torch._dynamo.reset()
                f = model if mode is None else torch.compile(model)
                def run(f=f):
                    with torch.autocast("cuda", dtype=torch.float16):
                        return f(*args)
                label = "pytorch fp16" if mode is None else "torch.compile fp16"
                report(label, run().float().cpu().numpy(), *timeit(run, sync, a.iters), dev_name)
        if "jax" in engines:
            import jax
            if jax.devices()[0].platform != "gpu":
                print(f"{a.name} jax: skipped, JAX has no GPU here ({jax.devices()[0].platform})")
            else:
                model = model.cpu()
                fn, jargs = bm.jax_model(a.name, model, feeds)
                jargs = tuple(jax.device_put(x, jax.devices()[0]) for x in jargs)
                run = lambda: fn(*jargs).block_until_ready()
                out = np.asarray(run())
                report("jax/xla", out, *timeit(run, lambda: None, a.iters), dev_name)
    if "ort" in engines:
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            print(f"{a.name} ort: skipped, no CUDAExecutionProvider")
        else:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            s = ort.InferenceSession(os.path.join(a.models, a.name + ".onnx"), so,
                                     providers=["CUDAExecutionProvider"])
            io = s.io_binding()
            for n, v in feeds.items():
                io.bind_ortvalue_input(n, ort.OrtValue.ortvalue_from_numpy(v, "cuda", 0))
            out_name = s.get_outputs()[0].name
            io.bind_output(out_name, "cuda")
            run = lambda: s.run_with_iobinding(io)
            run()
            out = io.copy_outputs_to_cpu()[0]
            report("onnxruntime", out, *timeit(run, io.synchronize_outputs, a.iters), dev_name)
    if a.json:
        with open(a.json, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
