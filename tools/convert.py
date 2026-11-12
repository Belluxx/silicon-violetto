from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch
from safetensors import safe_open

from common import MODEL, ROOT, load_model

N_LAYERS = 48
VE_LAYERS = set(range(1, 48, 3))
MUDD_SITES = (24, 47)
BF16 = torch.bfloat16
DATA_ALIGN = 16384
TENSOR_ALIGN = 256
DTYPES = {torch.float32: 0, BF16: 1, torch.int8: 2}
KINDS = ("qkv", "o", "gate_up", "down", "head")
INT8_DEFAULT = "qkv=64,o=64,gate_up=64,down=64,head=8"


def parse_int8(spec: str) -> dict[str, int]:
    """'qkv=64,head=32' -> {'qkv': 64, 'head': 32}. Groups are powers of two from 8 up
    that divide every k (1280 and 3328), so 8 to 256."""
    out = {}
    for item in filter(None, spec.split(",")):
        kind, group = item.split("=")
        g = int(group)
        if kind not in KINDS or g < 8 or g > 256 or g & (g - 1):
            raise SystemExit(f"bad int8 spec item {item!r}: kinds {KINDS}, groups 8..256, powers of two")
        out[kind] = g
    return out


def spec_tag(spec: dict[str, int]) -> str:
    """Directory-safe name of a spec, in KINDS order: 'qkv64_o64_gate_up64_down64_head32'."""
    return "_".join(f"{k}{spec[k]}" for k in KINDS if k in spec) or "bf16"


def quantize(w: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 over groups of `group` consecutive k: q in [-127, 127], one bf16 scale
    per group, absmax / 127 rounded to nearest. When the rounding goes down, the clamp moves
    the group's largest weight by at most a quarter step."""
    m, k = w.shape
    g = w.float().reshape(m, k // group, group)
    s = (g.abs().amax(-1) / 127).to(BF16)
    sf = s.float().unsqueeze(-1)
    q = torch.where(sf > 0, g / sf, 0.0).round().clamp(-127, 127).to(torch.int8)
    return q.reshape(m, k), s


def dequantize(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """fp32 weights of an int8 matrix (exact: an int8 times a bf16 fits in fp32)."""
    m, k = q.shape
    return (q.float().reshape(m, s.shape[1], -1) * s.float().unsqueeze(-1)).reshape(m, k)


def rope_frequency() -> torch.Tensor:
    """LimiteRotaryEmbedding._build_frequency, first 32 entries (pairs are repeated)."""
    return (1.0 / 1024.0) ** torch.linspace(0, 1, steps=32, dtype=torch.float32, device="cpu")


def build(int8: dict[str, int]) -> dict[str, torch.Tensor]:
    f = safe_open(MODEL / "model.safetensors", "pt")
    get = f.get_tensor
    out: dict[str, torch.Tensor] = {}

    def matrix(name: str, kind: str, w: torch.Tensor) -> None:
        if kind in int8:
            q, s = quantize(w, int8[kind])
            out[name + ".scale"] = s
            out[name] = q
        else:
            out[name] = w.contiguous()

    embed = get("model.embed_tokens.weight")
    out.update({
        "embed": embed,
        "value_embed": get("model.value_embeds.weight"),
        "rope_freq": rope_frequency(),
        "mudd.dense1": get("model.mudd.dense1").float(),
    })
    mix = torch.zeros(2, 2, 3, 33)
    for s, layer in enumerate(MUDD_SITES):
        for w, (d2, b) in enumerate((("dense2", "bias"), ("dense2_mlp", "bias_mlp"))):
            mix[s, w, :, :32] = get(f"model.mudd.{d2}")[layer].float()
            mix[s, w, :, 32] = get(f"model.mudd.{b}")[layer].float()
    out["mudd.mix"] = mix

    for i in range(N_LAYERS):
        p = f"model.layers.{i}."
        a = p + "self_attn."
        qkv = torch.cat([get(a + "q_proj.weight"), get(a + "k_proj.weight"), get(a + "v_proj.weight")])
        scales = [get(a + "qkv_scale").float().item(), get(a + "o_scale").float().item()]
        gate, up = get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight")
        lambdas = [get(p + k).float().item()
                   for k in ("resid_lambda_attn", "post_lambda_attn", "resid_lambda_mlp", "post_lambda_mlp")]
        L = f"L{i:02d}."
        matrix(L + "qkv", "qkv", qkv)
        matrix(L + "o", "o", get(a + "o_proj.weight"))
        matrix(L + "gate_up", "gate_up", torch.stack([gate, up], dim=1).reshape(-1, gate.shape[1]))
        matrix(L + "down", "down", get(p + "mlp.down_proj.weight"))
        out[L + "attn_gate"] = get(a + "attn_gate").float()
        if i in VE_LAYERS:
            out[L + "ve_gate"] = get(a + "ve_gate").float()[:2].contiguous()
        out[L + "xsa"] = torch.tanh(get(a + "xsa_alpha").float())
        out[L + "scalars"] = torch.tensor(scales + lambdas + [0.0, 0.0], dtype=torch.float32)
    if "head" in int8:
        matrix("head", "head", embed)
    for name, t in out.items():
        assert t.dtype in DTYPES, (name, t.dtype)
        assert t.dim() <= 4, name
    return out


def write(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    """The engine's weight file (engine/src/weights.rs reads it). Little endian:

        0   magic  b"VIOLETTO"
        8   u32    version (1)
        12  u32    number of tensors
        16  u64    offset of the data region (16 KiB aligned)
        24  u32    reserved
        28  u32    reserved
        32  records, 128 bytes each:
              name [64] (utf-8, NUL padded), u32 dtype (0 f32, 1 bf16, 2 int8), u32 ndim,
              u32 shape[4], u64 offset (from the data region), u64 nbytes, 24 reserved
        data region: tensors, each 256-byte aligned
    """
    records, blobs, offset = [], [], 0
    for name, t in tensors.items():
        offset = -(-offset // TENSOR_ALIGN) * TENSOR_ALIGN
        raw = t.contiguous().view(torch.uint8).numpy().tobytes()
        shape = list(t.shape) + [0] * (4 - t.dim())
        name_b = name.encode()
        assert len(name_b) < 64
        records.append(struct.pack("<64sII4IQQ24x", name_b, DTYPES[t.dtype], t.dim(), *shape, offset, len(raw)))
        blobs.append((offset, raw))
        offset += len(raw)
    header_len = 32 + 128 * len(records)
    data_offset = -(-header_len // DATA_ALIGN) * DATA_ALIGN
    total = data_offset + -(-offset // DATA_ALIGN) * DATA_ALIGN
    with open(path, "wb") as fh:
        fh.write(struct.pack("<8sIIQII", b"VIOLETTO", 1, len(records), data_offset, 0, 0))
        fh.writelines(records)
        for off, raw in blobs:
            fh.seek(data_offset + off)
            fh.write(raw)
        fh.truncate(total)


def read(path: Path) -> dict[str, torch.Tensor]:
    """Every tensor of a weight file, for the Python tooling."""
    by_code = {v: k for k, v in DTYPES.items()}
    data = path.read_bytes()
    magic, version, n, data_offset, _, _ = struct.unpack_from("<8sIIQII", data, 0)
    assert magic == b"VIOLETTO" and version == 1, path
    out = {}
    for i in range(n):
        name_b, dtype, ndim, *rest = struct.unpack_from("<64sII4IQQ24x", data, 32 + 128 * i)
        shape, (offset, nbytes) = rest[:ndim], rest[4:]
        raw = bytearray(data[data_offset + offset:data_offset + offset + nbytes])
        out[name_b.rstrip(b"\0").decode()] = torch.frombuffer(raw, dtype=by_code[dtype]).reshape(shape)
    return out


def int8_spec(tensors: dict[str, torch.Tensor]) -> dict[str, int]:
    """The --int8 spec a weight file was written with."""
    spec = {}
    for kind in KINDS:
        names = ["head"] if kind == "head" else [f"L{i:02d}.{kind}" for i in range(N_LAYERS)]
        groups = {tensors[n].shape[1] // tensors[n + ".scale"].shape[1] if n + ".scale" in tensors else 0
                  for n in names if n in tensors}
        assert len(groups) <= 1, f"{kind}: mixed formats across layers"
        if groups and (g := groups.pop()):
            spec[kind] = g
    return spec


def head_weight(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    """The head's weights in fp32, as the engine multiplies with them."""
    if "head" in tensors:
        return dequantize(tensors["head"], tensors["head.scale"])
    return tensors["embed"].float()


def verify(tensors: dict[str, torch.Tensor], int8: dict[str, int]) -> None:
    """Bitwise check of every tensor against the parameters of HF's own model, loaded in
    fp32 (so through HF's fused qkv_proj and gate_up_proj, not the checkpoint keys read by
    build()). An int8 matrix is checked against quantize() of HF's weights."""
    model = load_model("cpu", torch.float32)
    m = model.model
    bad, seen = 0, set()

    def check(name: str, ours: torch.Tensor, ref: torch.Tensor) -> None:
        nonlocal bad
        seen.add(name.split("[")[0])
        if ours.shape != ref.shape or not torch.equal(ours.float(), ref.float()):
            bad += 1
            diff = (ours.float() - ref.float()).abs().max().item() if ours.shape == ref.shape else float("nan")
            print(f"  MISMATCH {name}: shapes {tuple(ours.shape)} {tuple(ref.shape)}, max |diff| {diff:.3e}")

    def check_matrix(name: str, kind: str, ref: torch.Tensor) -> None:
        if kind in int8:
            q, s = quantize(ref, int8[kind])
            check(name + ".scale", tensors[name + ".scale"], s)
            ref = q
        check(name, tensors[name], ref)

    check("embed", tensors["embed"], m.embed_tokens.weight)
    if "head" in int8:
        check_matrix("head", "head", model.lm_head.weight)
    else:
        check("embed", tensors["embed"], model.lm_head.weight)  # tied head
    check("value_embed", tensors["value_embed"], m.value_embeds.weight)
    for i, layer in enumerate(m.layers):
        L, at = f"L{i:02d}.", layer.self_attn
        check_matrix(L + "qkv", "qkv", at.qkv_proj.weight)
        check_matrix(L + "o", "o", at.o_proj.weight)
        g, u = layer.mlp.gate_up_proj.weight.chunk(2)
        check_matrix(L + "gate_up", "gate_up", torch.stack([g, u], 1).reshape(-1, g.shape[1]))
        check_matrix(L + "down", "down", layer.mlp.down_proj.weight)
        check(L + "attn_gate", tensors[L + "attn_gate"], at.attn_gate)
        if at.has_ve:
            check(L + "ve_gate", tensors[L + "ve_gate"], at.ve_gate[: at.num_kv_heads])
        check(L + "xsa", tensors[L + "xsa"], torch.tanh(at.xsa_alpha))
        scalars = torch.stack([at.qkv_scale, at.o_scale, layer.resid_lambda_attn, layer.post_lambda_attn,
                               layer.resid_lambda_mlp, layer.post_lambda_mlp]).reshape(6)
        check(L + "scalars", tensors[L + "scalars"], torch.cat([scalars, torch.zeros(2)]))
    check("mudd.dense1", tensors["mudd.dense1"], m.mudd.dense1)
    for s, layer in enumerate(MUDD_SITES):
        check(f"mudd.mix[{layer}].dense2", tensors["mudd.mix"][s, 0, :, :32], m.mudd.dense2[layer])
        check(f"mudd.mix[{layer}].bias", tensors["mudd.mix"][s, 0, :, 32], m.mudd.bias[layer])
        check(f"mudd.mix[{layer}].dense2_mlp", tensors["mudd.mix"][s, 1, :, :32], m.mudd.dense2_mlp[layer])
        check(f"mudd.mix[{layer}].bias_mlp", tensors["mudd.mix"][s, 1, :, 32], m.mudd.bias_mlp[layer])
    check("rope_freq", tensors["rope_freq"].repeat_interleave(2), m.rotary_emb.frequency[:64])
    unchecked = set(tensors) - seen
    if unchecked:
        bad += 1
        print(f"  UNCHECKED: {sorted(unchecked)}")
    print("verify: every tensor bit-identical to HF's fp32 model" if bad == 0 else f"verify: {bad} mismatches")
    if bad:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", nargs="?", const=INT8_DEFAULT, default="", metavar="SPEC",
                    help=f"matrix kinds to quantize, kind=group for any of {', '.join(KINDS)}; "
                         f"the rest stay bf16 (default {INT8_DEFAULT})")
    ap.add_argument("--out", type=Path, help="default models/limite-bf16.vlt, or limite-int8.vlt with --int8")
    ap.add_argument("--verify", action="store_true", help="compare every tensor against HF's fp32 model")
    args = ap.parse_args()
    int8 = parse_int8(args.int8)
    out = args.out or ROOT / "models" / ("limite-int8.vlt" if int8 else "limite-bf16.vlt")
    tensors = build(int8)
    if args.verify:
        verify(tensors, int8)
    write(out, tensors)
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB, {len(tensors)} tensors, int8: {spec_tag(int8)})")


if __name__ == "__main__":
    main()
