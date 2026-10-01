| Model | kiln | onnxruntime | pytorch | torch.compile | jax/xla |
|---|---|---|---|---|---|
| MLP (4 blocks, 64 x 512) | **2.7 ms** | 2.9 ms | 5.3 ms | 4.8 ms | 4.3 ms |
| BERT bge-small (8 x 128 tokens) | **126.3 ms** | 145.8 ms | 179.8 ms | 133.1 ms | 170.3 ms |
| SmolLM2-135M (128 tokens) | **102.0 ms** | 110.5 ms | 137.6 ms | 109.2 ms | 126.4 ms |

Max |difference| from the PyTorch reference:
- mlp: kiln 2.1e-06, onnxruntime 9.5e-07, pytorch 0.0e+00, torch.compile 9.5e-07, jax/xla 1.3e-06
- bert: kiln 2.7e-07, onnxruntime 2.1e-07, pytorch 0.0e+00, torch.compile 2.2e-07, jax/xla 1.9e-07
- llama: kiln 2.1e-04, onnxruntime 9.5e-05, pytorch 0.0e+00, torch.compile 5.3e-05, jax/xla 8.7e-05

| Model | kiln (all) | --no-fusion | --no-epilogue | --no-rows | --no-inline | --no-fold | --no-tune |
|---|---|---|---|---|---|---|---|
| mlp | 2.7 ms | 3.2 ms (1.18x) | 2.6 ms (0.95x) | 2.8 ms (1.02x) | 2.6 ms (0.95x) | 2.7 ms (1.00x) | 3.3 ms (1.21x) |
| bert | 126.3 ms | 151.1 ms (1.20x) | 129.0 ms (1.02x) | 127.7 ms (1.01x) | 127.4 ms (1.01x) | 148.2 ms (1.17x) | 165.5 ms (1.31x) |
| llama | 102.0 ms | 110.6 ms (1.08x) | 97.9 ms (0.96x) | 101.6 ms (1.00x) | 106.4 ms (1.04x) | 102.1 ms (1.00x) | 127.8 ms (1.25x) |

- mlp: 12 kernels; arena 0.8 MB for 3.3 MB of intermediates
- bert: 134 kernels; arena 16.2 MB for 381.2 MB of intermediates
- llama: 422 kernels; arena 25.5 MB for 179.1 MB of intermediates

Samples per engine: bert/jax/xla 60, bert/kiln 60, bert/kiln --no-epilogue 60, bert/kiln --no-fold 60, bert/kiln --no-fusion 60, bert/kiln --no-inline 60, bert/kiln --no-rows 60, bert/kiln --no-tune 60, bert/onnxruntime 60, bert/pytorch 60, bert/torch.compile 60, llama/jax/xla 60, llama/kiln 60, llama/kiln --no-epilogue 60, llama/kiln --no-fold 60, llama/kiln --no-fusion 60, llama/kiln --no-inline 60, llama/kiln --no-rows 60, llama/kiln --no-tune 60, llama/onnxruntime 60, llama/pytorch 60, llama/torch.compile 60, mlp/jax/xla 300, mlp/kiln 300, mlp/kiln --no-epilogue 300, mlp/kiln --no-fold 300, mlp/kiln --no-fusion 300, mlp/kiln --no-inline 300, mlp/kiln --no-rows 300, mlp/kiln --no-tune 300, mlp/onnxruntime 300, mlp/pytorch 300, mlp/torch.compile 300
