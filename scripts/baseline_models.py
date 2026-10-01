"""The benchmark models for the baselines: PyTorch modules with the same
weights as the exported ONNX files, and the same networks written in JAX
(jit-compiled by XLA) over those weights."""

import os

import numpy as np
import torch

MODELS = os.environ.get("FERROLM_MODELS", "/home/user/ferrolm/models")


def torch_model(name, feeds):
    import export_models as em  # seeds torch with 0 on import, as the export did
    if name == "mlp":
        m = em.Mlp().eval()
        return m, (torch.from_numpy(feeds["x"]),)
    if name == "bert":
        m = em.Bert(os.path.join(MODELS, "bge-small")).eval()
        return m, (torch.from_numpy(feeds["input_ids"]), torch.from_numpy(feeds["attention_mask"]))
    if name == "llama":
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(os.path.join(MODELS, "smollm2-135m"), torch_dtype=torch.float32,
                                                  attn_implementation="eager").eval()
        m = em.Llama(hf, feeds["input_ids"].shape[1]).eval()
        return m, (torch.from_numpy(feeds["input_ids"]),)
    raise ValueError(name)


def jax_model(name, model, feeds):
    import jax
    import jax.numpy as jnp
    p = {k: jnp.asarray(v.detach().numpy()) for k, v in model.state_dict().items()}

    def ln(x, w, b, eps):
        m = x.mean(-1, keepdims=True)
        v = ((x - m) ** 2).mean(-1, keepdims=True)
        return (x - m) / jnp.sqrt(v + eps) * w + b

    gelu = lambda x: jax.nn.gelu(x, approximate=False)
    lin = lambda x, pre: x @ p[pre + ".weight"].T + p[pre + ".bias"]

    if name == "mlp":
        def f(x):
            for i in range(len(model.blocks)):
                b = f"blocks.{i}."
                h = ln(x, p[b + "ln.weight"], p[b + "ln.bias"], 1e-5)
                x = x + lin(gelu(lin(h, b + "up")), b + "down")
            return x
        return jax.jit(f), (jnp.asarray(feeds["x"]),)

    if name == "bert":
        c = model.m.config
        nh, hd = c.num_attention_heads, c.hidden_size // c.num_attention_heads
        eps = c.layer_norm_eps

        def f(ids, mask):
            b, t = ids.shape
            e = "m.embeddings."
            x = p[e + "word_embeddings.weight"][ids] + p[e + "position_embeddings.weight"][:t] + p[e + "token_type_embeddings.weight"][0]
            x = ln(x, p[e + "LayerNorm.weight"], p[e + "LayerNorm.bias"], eps)
            bias = (1.0 - mask[:, None, None, :].astype(jnp.float32)) * jnp.finfo(jnp.float32).min
            for i in range(c.num_hidden_layers):
                l = f"m.encoder.layer.{i}."
                heads = lambda y: y.reshape(b, t, nh, hd).transpose(0, 2, 1, 3)
                q, k, v = (heads(lin(x, l + "attention.self." + n)) for n in ("query", "key", "value"))
                s = q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(float(hd)) + bias
                o = (jax.nn.softmax(s, -1) @ v).transpose(0, 2, 1, 3).reshape(b, t, -1)
                x = ln(x + lin(o, l + "attention.output.dense"), p[l + "attention.output.LayerNorm.weight"],
                       p[l + "attention.output.LayerNorm.bias"], eps)
                h = gelu(lin(x, l + "intermediate.dense"))
                x = ln(x + lin(h, l + "output.dense"), p[l + "output.LayerNorm.weight"], p[l + "output.LayerNorm.bias"], eps)
            cls = x[:, 0]
            return cls / jnp.linalg.norm(cls, axis=-1, keepdims=True)
        return jax.jit(f), (jnp.asarray(feeds["input_ids"]), jnp.asarray(feeds["attention_mask"]))

    if name == "llama":
        c = model.c
        nh, kv = c.num_attention_heads, c.num_key_value_heads
        hd = c.hidden_size // nh
        eps = c.rms_norm_eps
        cos, sin, cmask = p["cos"], p["sin"], p["mask"]
        rms = lambda x, w: w * (x * jax.lax.rsqrt((x * x).mean(-1, keepdims=True) + eps))
        mm = lambda x, n: x @ p[n + ".weight"].T

        def rope(x):
            h = hd // 2
            return x * cos + jnp.concatenate([-x[..., h:], x[..., :h]], -1) * sin

        def f(ids):
            b, t = ids.shape
            x = p["hf.model.embed_tokens.weight"][ids]
            for i in range(c.num_hidden_layers):
                l = f"hf.model.layers.{i}."
                h = rms(x, p[l + "input_layernorm.weight"])
                q = mm(h, l + "self_attn.q_proj").reshape(b, t, nh, hd).transpose(0, 2, 1, 3)
                k = mm(h, l + "self_attn.k_proj").reshape(b, t, kv, hd).transpose(0, 2, 1, 3)
                v = mm(h, l + "self_attn.v_proj").reshape(b, t, kv, hd).transpose(0, 2, 1, 3)
                q, k = rope(q), rope(k)
                k = jnp.repeat(k, nh // kv, axis=1)
                v = jnp.repeat(v, nh // kv, axis=1)
                s = q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(float(hd)) + cmask
                o = (jax.nn.softmax(s, -1) @ v).transpose(0, 2, 1, 3).reshape(b, t, -1)
                x = x + mm(o, l + "self_attn.o_proj")
                h = rms(x, p[l + "post_attention_layernorm.weight"])
                x = x + mm(jax.nn.silu(mm(h, l + "mlp.gate_proj")) * mm(h, l + "mlp.up_proj"), l + "mlp.down_proj")
            x = rms(x, p["hf.model.norm.weight"])
            return x @ p["hf.lm_head.weight"].T
        return jax.jit(f), (jnp.asarray(feeds["input_ids"]),)
    raise ValueError(name)
