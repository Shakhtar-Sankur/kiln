"""Exports the benchmark models to ONNX with static shapes, plus a reference
input and PyTorch's output for each (float32), for kiln's tests.

Usage: python scripts/export_models.py OUT_DIR [mlp|bert|llama ...]

  mlp    a 4-layer MLP block (LayerNorm, Linear, GELU, residual), batch 64
  bert   BAAI/bge-small-en-v1.5 (needs models/bge-small), batch 8 x 128 tokens
  llama  SmolLM2-135M (needs models/smollm2-135m), 1 x 128 tokens -> logits
  tiny   small random-weight versions of all three (tests/fixtures)

Each model writes NAME.onnx and NAME.ref (see write_ref) to OUT_DIR.
"""

import json
import os
import struct
import sys

import numpy as np
import torch

torch.manual_seed(0)


def write_ref(path, inputs, outputs):
    """kiln reference file: b"KILNREF1", u32 header length, a JSON header
    listing each tensor's name, dtype and shape, then the raw little-endian
    data of every tensor in order."""
    tensors = [(n, t) for n, t in inputs] + [(n, t) for n, t in outputs]
    meta = {"inputs": [], "outputs": []}
    blobs = []
    for kind, group in (("inputs", inputs), ("outputs", outputs)):
        for name, t in group:
            a = t.detach().cpu().numpy()
            dtype = {np.float32: "f32", np.int64: "i64"}[a.dtype.type]
            meta[kind].append({"name": name, "dtype": dtype, "shape": list(a.shape)})
            blobs.append(np.ascontiguousarray(a).tobytes())
    head = json.dumps(meta).encode()
    with open(path, "wb") as f:
        f.write(b"KILNREF1" + struct.pack("<I", len(head)) + head)
        for b in blobs:
            f.write(b)


def export(model, args, names, out_names, path):
    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, args, path, input_names=names, output_names=out_names,
                          opset_version=17, do_constant_folding=True, dynamo=False)


class Block(torch.nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln = torch.nn.LayerNorm(d)
        self.up = torch.nn.Linear(d, h)
        self.down = torch.nn.Linear(h, d)

    def forward(self, x):
        return x + self.down(torch.nn.functional.gelu(self.up(self.ln(x))))


class Mlp(torch.nn.Module):
    def __init__(self, d=512, h=2048, n=4):
        super().__init__()
        self.blocks = torch.nn.ModuleList(Block(d, h) for _ in range(n))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def mlp(out):
    m = Mlp()
    x = torch.randn(64, 512)
    export(m, (x,), ["x"], ["y"], os.path.join(out, "mlp.onnx"))
    with torch.no_grad():
        write_ref(os.path.join(out, "mlp.ref"), [("x", x)], [("y", m(x))])


class Bert(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        from transformers import BertModel
        self.m = BertModel.from_pretrained(path, attn_implementation="eager")

    def forward(self, ids, mask):
        h = self.m(input_ids=ids, attention_mask=mask).last_hidden_state
        cls = h[:, 0]
        return cls / cls.norm(dim=-1, keepdim=True)


def bert(out, path="models/bge-small"):
    m = Bert(path)
    ids = torch.randint(1000, 20000, (8, 128))
    ids[:, 0] = 101
    mask = torch.ones(8, 128, dtype=torch.int64)
    mask[4:, 100:] = 0  # some padded rows
    export(m, (ids, mask), ["input_ids", "attention_mask"], ["embedding"], os.path.join(out, "bert.onnx"))
    with torch.no_grad():
        write_ref(os.path.join(out, "bert.ref"), [("input_ids", ids), ("attention_mask", mask)],
                  [("embedding", m(ids, mask))])


class Llama(torch.nn.Module):
    """A plain Llama forward pass (RMSNorm, rotary embeddings, grouped-query
    attention with a causal mask, SwiGLU) over SmolLM2's weights, written
    out so that it exports cleanly; checked against Hugging Face below."""

    def __init__(self, hf, seq):
        super().__init__()
        c = hf.config
        self.c = c
        self.hf = hf
        self.hd = c.hidden_size // c.num_attention_heads
        theta = getattr(c, "rope_theta", None) or (getattr(c, "rope_parameters", None) or {}).get("rope_theta", 10000.0)
        inv = 1.0 / (theta ** (torch.arange(0, self.hd, 2).float() / self.hd))
        f = torch.outer(torch.arange(seq).float(), inv)
        emb = torch.cat([f, f], dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())
        self.register_buffer("mask", torch.full((seq, seq), float("-inf")).triu(1))

    def norm(self, x, w):
        return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.c.rms_norm_eps))

    def rope(self, x):
        h = self.hd // 2
        rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
        return x * self.cos + rot * self.sin

    def forward(self, ids):
        c, m = self.c, self.hf.model
        b, t = ids.shape
        nh, kv = c.num_attention_heads, c.num_key_value_heads
        x = m.embed_tokens(ids)
        for l in m.layers:
            a = l.self_attn
            h = self.norm(x, l.input_layernorm.weight)
            q = a.q_proj(h).view(b, t, nh, self.hd).transpose(1, 2)
            k = a.k_proj(h).view(b, t, kv, self.hd).transpose(1, 2)
            v = a.v_proj(h).view(b, t, kv, self.hd).transpose(1, 2)
            q, k = self.rope(q), self.rope(k)
            k = k.repeat_interleave(nh // kv, dim=1)
            v = v.repeat_interleave(nh // kv, dim=1)
            s = q @ k.transpose(-1, -2) / self.hd ** 0.5 + self.mask
            o = (s.softmax(-1) @ v).transpose(1, 2).reshape(b, t, -1)
            x = x + a.o_proj(o)
            h = self.norm(x, l.post_attention_layernorm.weight)
            mm = l.mlp
            x = x + mm.down_proj(torch.nn.functional.silu(mm.gate_proj(h)) * mm.up_proj(h))
        return self.hf.lm_head(self.norm(x, m.norm.weight))


def llama(out, path="models/smollm2-135m", seq=128):
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="eager").eval()
    m = Llama(hf, seq).eval()
    ids = torch.randint(0, hf.config.vocab_size, (1, seq))
    with torch.no_grad():
        want = hf(input_ids=ids, use_cache=False).logits
        got = m(ids)
    err = (got - want).abs().max().item()
    print(f"llama: plain forward vs Hugging Face, max |difference| {err:.2e}")
    assert err < 1e-3, err
    export(m, (ids,), ["input_ids"], ["logits"], os.path.join(out, "llama.onnx"))
    write_ref(os.path.join(out, "llama.ref"), [("input_ids", ids)], [("logits", got)])


def tiny(out):
    """Small random-weight versions of the three models, for tests and CI."""
    from transformers import BertConfig, BertModel, LlamaConfig, LlamaForCausalLM

    m = Mlp(d=48, h=96, n=2)
    x = torch.randn(10, 48)
    export(m, (x,), ["x"], ["y"], os.path.join(out, "mlp_tiny.onnx"))
    with torch.no_grad():
        write_ref(os.path.join(out, "mlp_tiny.ref"), [("x", x)], [("y", m(x))])

    cfg = BertConfig(vocab_size=200, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                     intermediate_size=64, max_position_embeddings=64, attn_implementation="eager")
    bm = BertModel(cfg, add_pooling_layer=False).eval()

    class TinyBert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = bm

        def forward(self, ids, mask):
            h = self.m(input_ids=ids, attention_mask=mask).last_hidden_state
            cls = h[:, 0]
            return cls / cls.norm(dim=-1, keepdim=True)

    tb = TinyBert().eval()
    ids = torch.randint(0, 200, (3, 17))
    mask = torch.ones(3, 17, dtype=torch.int64)
    mask[1, 12:] = 0
    export(tb, (ids, mask), ["input_ids", "attention_mask"], ["embedding"], os.path.join(out, "bert_tiny.onnx"))
    with torch.no_grad():
        write_ref(os.path.join(out, "bert_tiny.ref"), [("input_ids", ids), ("attention_mask", mask)],
                  [("embedding", tb(ids, mask))])

    lc = LlamaConfig(vocab_size=300, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                     rope_theta=10000.0, tie_word_embeddings=True, attn_implementation="eager")
    hf = LlamaForCausalLM(lc).eval()
    lm = Llama(hf, 19).eval()
    ids = torch.randint(0, 300, (1, 19))
    with torch.no_grad():
        want = hf(input_ids=ids, use_cache=False).logits
        got = lm(ids)
    assert (got - want).abs().max().item() < 1e-4
    export(lm, (ids,), ["input_ids"], ["logits"], os.path.join(out, "llama_tiny.onnx"))
    write_ref(os.path.join(out, "llama_tiny.ref"), [("input_ids", ids)], [("logits", got)])


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for name in sys.argv[2:] or ["mlp", "bert", "llama"]:
        globals()[name](out)
        print("exported", name)
