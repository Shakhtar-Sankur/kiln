# kiln

An ML compiler written from scratch in Rust, with no dependencies. It reads
a model exported from PyTorch as ONNX, optimizes the graph, fuses it into a
small number of kernels, generates SIMD C code for them, compiles that code
with the system C compiler, auto-tunes the matrix multiplies on the machine
it runs on, and executes the result: the pipeline of XLA, TVM or
torch.compile's Inductor, end to end.

On real models it is checked against PyTorch, and on a 4-vCPU machine it
is faster than ONNX Runtime, PyTorch and JAX/XLA, and level with
torch.compile:

@@HEADLINE@@

## Results

![Latency of kiln, torch.compile, ONNX Runtime, JAX/XLA and PyTorch](docs/latency.svg)

@@TABLE@@

All runs: `bench/run.sh` (rounds interleave every engine so that slow
periods of the machine hit all of them; the tables are medians of every
timing, raw data in `bench/results/runs.jsonl`). Machine: 4 vCPUs of an
Intel Xeon (Cascade Lake, AVX-512), 15 GB RAM, every engine limited to 4
threads. Models in float32; the same inputs for every engine, whose output
is compared with PyTorch's before it is timed. Baselines: ONNX Runtime
1.30 (all graph optimizations), PyTorch 2.14 eager and `torch.compile`
(Inductor, C++ backend), JAX 0.10 (`jax.jit`, XLA CPU) running the same
networks written in JAX over the same weights.

### What each optimization is worth

![Latency with each optimization switched off](docs/ablations.svg)

@@ABLATIONS@@

### Compilation

@@COMPILE@@

## Correctness

| Check | Against | Result |
|---|---|---|
@@CORRECTNESS@@

## How it works

**Import.** A protobuf wire-format decoder reads the ONNX file into kiln's
graph IR: named values with types, static shapes and, where known,
constant contents.

**Graph passes.** Shape and type inference runs in topological order
together with constant folding: any node whose inputs are all known, and
any `Shape` of a value with a known shape, is evaluated at compile time by
the reference interpreter. With static input shapes, the shape arithmetic
and mask construction exporters emit disappear (1,593 of SmolLM2's 3,402
nodes). Canonicalization then rewrites the graph into a small core set:
`Gemm` becomes `MatMul` and `Add`; `Softmax`, `LayerNormalization` and
`ReduceL2` are decomposed into last-axis reductions and elementwise
operations (fusion puts them back together as single kernels, as XLA
does); `Flatten`, `Squeeze` and `Unsqueeze` become `Reshape`. Dead code is
removed.

**Fusion.** Every value is either a *root*, written to memory, or *inline*,
recomputed inside whichever kernel reads it. Roots are matrix multiplies,
reductions, graph outputs, values read more than once, and values a matmul
reads. Data movement (reshape, transpose, slice, concat, tile, expand) is
never materialized: it becomes index arithmetic in its readers. Kernels:

- a **matmul with its epilogue**: the chain of elementwise roots that alone
  consume the product (bias, GELU, residual add, attention scale and mask)
  is computed in registers, intermediate links as temporaries, before a
  single store;
- **row kernels**: consecutive elementwise operations and last-axis
  reductions over the same rows become one kernel, one row per task, with
  row-local scalars and buffers for values nothing else reads; softmax,
  layer norm and RMSNorm are each a single pass;
- **batch folding**: a matmul whose weight is shared across a batch becomes
  one tall matmul when its operands stay affine in the folded row index;
- host steps for the little integer and boolean work left (attention-mask
  construction, token lookups).

**Index arithmetic.** Kernel IR indices are affine forms over loop
variables plus floor-division and remainder atoms, kept canonical by a
simplifier that knows every variable's range. It splits divisions when the
remainder provably stays in range, factors through common divisors
(grouped-query attention's `(8192·h + j + 64·k) / 24576` becomes `h / 3`),
and recombines `a·(x / d) + b·(x mod d)` into `b·x` when `a = b·d`.
Reshapes and transposes therefore compile to plain strides, and the code
generator can read off whether each access is contiguous, broadcast or a
gather.

**Code generation.** Each kernel becomes a C function over a range of
tasks, using GCC's vector extensions (16-float vectors with AVX-512, 8
with AVX2). Matmuls use an MR × NR register tile of vector accumulators
fed by broadcast loads of A and vector loads of B, which is packed into
NR-wide panels: at compile time for constant weights, per task for
activations (so transposed and grouped operands cost one pass). K is
optionally blocked, with partial sums kept in the output buffer and the
epilogue run on the last block. Row kernels vectorize the inner dimension
with scalar tails and per-chunk uniform selects. `exp` and `erf` are
vectorized polynomials; `exp` returns exactly zero below the float range
(a masked `-inf` would otherwise become a subnormal, and subnormal
arithmetic is a hundred times slower). Identical kernels in different
layers share one function, so SmolLM2's 422 kernels compile from 13.

**Auto-tuning.** For each distinct matmul shape the tuner generates
variants (register tiles up to 12 × 64, K blocks, splitting the work by
rows, columns or both, software prefetch distances), compiles them in
parallel, and times each on that shape with the kernel's real epilogue.
Weights come in enough copies to exceed the last-level cache and are
rotated between runs, so each measurement reads them cold from memory, as
inference does. The fastest schedule is cached per shape, vector width,
thread count and CPU model.

**Runtime.** The generated C is compiled once (`cc -O3 -march=native`),
cached by content hash, and loaded with `dlopen`. A memory planner gives
every intermediate a lifetime in steps and packs them into one arena,
greedy by size: BERT's 381 MB of intermediates fit in 16 MB. A persistent
thread pool runs each kernel's tasks.

## Usage

```sh
cargo build --release

# Export the benchmark models (PyTorch, transformers) and their reference outputs
python scripts/export_models.py models mlp bert llama

# Compile, check against the reference, time; --tune searches matmul schedules (cached)
./target/release/kiln run models/bert.onnx models/bert.ref --tune --iters 20 --profile

# Look inside: the optimized graph, the fusion plan
./target/release/kiln opt models/llama.onnx models/llama.ref
./target/release/kiln plan tests/fixtures/llama_tiny.onnx tests/fixtures/llama_tiny.ref --dump

# The benchmark of this README
bench/run.sh 5 python
```

`--no-epilogue`, `--no-rows`, `--no-inline`, `--no-fold`, `--no-fusion`
and `--no-tune` switch single optimizations off. Generated C and tuning
results live in `~/.cache/kiln` (`KILN_CACHE`); `KILN_VEC=8` forces
AVX2-width vectors.

## Scope and limitations

- CPU only, float32, static shapes (one compilation per input shape, as
  XLA does). The operators covered are those of transformer and MLP
  models exported from PyTorch; convolutions are not supported.
- Integer and boolean operations run in the interpreter, not compiled
  code (they are a few percent of BERT's time, for its attention mask).
- Kernels come out of a fixed set of templates (matmul, row kernels); the
  tuner searches their parameters, not arbitrary loop nests.
- The benchmarks come from one 4-vCPU virtual machine; absolute numbers
  depend on the CPU, and run-to-run variation here is about ±10%, which
  is why every engine is measured many times, interleaved.

## Layout

| Path | What |
|---|---|
| `src/proto.rs`, `src/onnx.rs` | protobuf decoding, ONNX import |
| `src/graph.rs`, `src/tensor.rs` | graph IR, tensors |
| `src/interp.rs` | reference interpreter (oracle, constant folding, host steps) |
| `src/passes.rs` | shape inference, constant folding, canonicalization, DCE |
| `src/ir.rs` | kernel IR: affine indices with div/mod simplification, value expressions |
| `src/fuse.rs` | fusion planner |
| `src/eval.rs` | evaluates a plan from the IR (independent of code generation) |
| `src/codegen.rs` | C code generation |
| `src/jit.rs` | compiling and loading generated code |
| `src/tune.rs` | matmul auto-tuner |
| `src/runtime.rs`, `src/pool.rs` | memory planner, executor, thread pool |
| `tests/` | end-to-end tests on tiny models; fixtures in `tests/fixtures` |
| `scripts/` | model export, baselines (ONNX Runtime, PyTorch, torch.compile, JAX), charts |
| `bench/run.sh` | reproduces every benchmark in this README |

## License

Apache-2.0
