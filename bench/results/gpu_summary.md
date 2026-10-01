GPU: Tesla T4 (sm_75)

fp32, median latency

| Model | kiln | torch.compile | torch.compile (CUDA graphs) | onnxruntime | jax/xla | pytorch |
|---|---|---|---|---|---|---|
| MLP (4 blocks, 64 x 512) | 0.73 ms | 1.08 ms | 0.93 ms | **0.67 ms** | 0.93 ms | 0.95 ms |
| BERT bge-small (8 x 128 tokens) | 20.86 ms | 17.35 ms | 17.41 ms | 17.80 ms | **16.67 ms** | 20.01 ms |
| SmolLM2-135M (128 tokens) | 19.45 ms | 16.02 ms | 15.50 ms | 14.92 ms | **13.55 ms** | 30.48 ms |

fp16 matmul operands, fp32 accumulation (kiln --half; torch.autocast), median latency

| Model | kiln fp16 | torch.compile fp16 | pytorch fp16 |
|---|---|---|---|
| MLP (4 blocks, 64 x 512) | **0.30 ms** | 1.10 ms | 1.77 ms |
| BERT bge-small (8 x 128 tokens) | 9.54 ms | **7.13 ms** | 14.36 ms |
| SmolLM2-135M (128 tokens) | **7.94 ms** | 13.13 ms | 49.38 ms |

Max |difference| from the PyTorch (CPU, fp32) reference:
- mlp: kiln 2.1e-06, kiln fp16 7.3e-04, torch.compile 9.5e-07, torch.compile (CUDA graphs) 9.5e-07, onnxruntime 9.5e-07, jax/xla 1.2e-06, pytorch 1.1e-06, torch.compile fp16 1.2e-03, pytorch fp16 1.2e-03
- bert: kiln 1.8e-07, kiln fp16 1.2e-04, torch.compile 1.9e-07, torch.compile (CUDA graphs) 1.9e-07, onnxruntime 2.1e-07, jax/xla 2.4e-07, pytorch 2.7e-07, torch.compile fp16 1.5e-04, pytorch fp16 1.9e-04
- llama: kiln 2.0e-04, kiln fp16 7.7e-02, torch.compile 9.2e-05, torch.compile (CUDA graphs) 9.2e-05, onnxruntime 9.5e-05, jax/xla 1.2e-04, pytorch 1.5e-04, torch.compile fp16 1.7e-01, pytorch fp16 1.2e-01

| Model | kiln (all) | --no-attention | --no-graphs | --no-tune | --no-fusion |
|---|---|---|---|---|---|
| mlp | 0.73 ms | 0.71 ms (0.97x) | 0.72 ms (0.98x) | 1.25 ms (1.70x) | 0.85 ms (1.15x) |
| bert | 20.86 ms | 20.73 ms (0.99x) | 20.78 ms (1.00x) | 23.37 ms (1.12x) | 29.92 ms (1.43x) |
| llama | 19.45 ms | 19.68 ms (1.01x) | 20.10 ms (1.03x) | 28.72 ms (1.48x) | 21.56 ms (1.11x) |

Timings per cell: bert/jax/xla 150, bert/kiln 150, bert/kiln --no-attention 150, bert/kiln --no-fusion 150, bert/kiln --no-graphs 150, bert/kiln --no-tune 150, bert/kiln fp16 150, bert/onnxruntime 150, bert/pytorch 150, bert/pytorch fp16 150, bert/torch.compile 150, bert/torch.compile (CUDA graphs) 150, bert/torch.compile fp16 150, llama/jax/xla 150, llama/kiln 150, llama/kiln --no-attention 150, llama/kiln --no-fusion 150, llama/kiln --no-graphs 150, llama/kiln --no-tune 150, llama/kiln fp16 150, llama/onnxruntime 150, llama/pytorch 150, llama/pytorch fp16 150, llama/torch.compile 150, llama/torch.compile (CUDA graphs) 150, llama/torch.compile fp16 150, mlp/jax/xla 600, mlp/kiln 600, mlp/kiln --no-attention 600, mlp/kiln --no-fusion 600, mlp/kiln --no-graphs 600, mlp/kiln --no-tune 600, mlp/kiln fp16 600, mlp/onnxruntime 600, mlp/pytorch 600, mlp/pytorch fp16 600, mlp/torch.compile 600, mlp/torch.compile (CUDA graphs) 600, mlp/torch.compile fp16 600
