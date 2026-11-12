from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "limite-1b-violetto"  # the HF checkpoint


def load_model(device: str, dtype):
    """The HF implementation. HF casts the MUDD constants to bf16 at eval(), which breaks fp32
    activations, so in float32 they're put back in fp32."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=dtype, attn_implementation="sdpa")
    model = model.to(device).eval()  # eval() after the move: it derives the folded inference constants
    if dtype == torch.float32:
        mudd = model.model.mudd
        for name in ("dense1", "dense2", "bias", "dense2_mlp", "bias_mlp"):
            setattr(mudd, f"_inference_{name}", getattr(mudd, name).detach().float())
    return model
