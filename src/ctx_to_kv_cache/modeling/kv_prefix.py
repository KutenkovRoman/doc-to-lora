from __future__ import annotations

import torch
from torch import Tensor

from transformers import DynamicCache


def _rotate_half(x: Tensor) -> Tensor:
    """NeoX-style half rotation (identical across Qwen3/Llama/Mistral)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _get_layers(base_model: torch.nn.Module):
    """Return (transformer_module_with_rotary, decoder_layers_list)."""
    m = base_model
    while not hasattr(m, "layers"):
        if not hasattr(m, "model"):
            raise ValueError(
                f"Cannot locate decoder layers on {type(base_model).__name__}"
            )
        m = m.model
    return m, list(m.layers)


@torch.no_grad()
def _prefix_position_embeddings(transformer, ref: Tensor, p_eff: int):
    """cos/sin for prefix positions 0..p_eff-1 via the model's own rotary.

    `transformer.rotary_emb` is `@torch.no_grad()` internally already; the
    rotation itself (applied in :func:`build_kv_prefix_cache`) stays in the
    autograd graph because it only multiplies the grad-tracked K by these
    constant cos/sin tensors.
    """
    pos = torch.arange(p_eff, device=ref.device).unsqueeze(0)  # (1, P_eff)
    cos, sin = transformer.rotary_emb(ref, pos)  # (1, P_eff, head_dim)
    return cos.unsqueeze(1), sin.unsqueeze(1)  # (1, 1, P_eff, head_dim)


def build_kv_prefix_cache(
    base_model: torch.nn.Module,
    x_e: Tensor,
    layer_indices: tuple[int, ...] | None = None,
) -> DynamicCache:
    """Build a `DynamicCache` holding per-layer virtual K/V from `X_e`.

    Args:
        base_model: the (frozen) causal LM, e.g. Qwen3ForCausalLM.
        x_e: per-layer virtual hidden rows. Shape `(B, L_sel, P_eff, D)` where
            `L_sel == len(layer_indices)` (or `== num_layers` when None).
        layer_indices: absolute base-layer indices that receive a prefix.
            `None` = all layers (legacy behaviour). When a subset is given,
            unselected layers get a **0-length** K/V entry, so their attention
            sees only the real tokens.

    Returns:
        A `DynamicCache` with one entry per base layer. Selected layers hold
        `(B, H_kv, P_eff, head_dim)` keys RoPE-rotated at positions
        `0..P_eff-1`; unselected layers hold `(B, H_kv, 0, head_dim)`.
        Differentiable w.r.t. `x_e`.
    """
    if x_e.dim() != 4:
        raise ValueError(f"x_e must be (B, L, P_eff, D); got {tuple(x_e.shape)}")
    transformer, layers = _get_layers(base_model)
    # The hypernet emits fp32 (LayerNorm runs in fp32 even under autocast);
    # the frozen base projections are bf16. Cast once — differentiable, so
    # grads still flow back to the hypernet (mirrors the LoRA path).
    proj_dtype = layers[0].self_attn.k_proj.weight.dtype
    x_e = x_e.to(proj_dtype)
    B, L_sel, P_eff, _ = x_e.shape

    if layer_indices is None:
        sel = list(range(len(layers)))
    else:
        sel = list(layer_indices)
        if any(not (0 <= i < len(layers)) for i in sel):
            raise ValueError(
                f"layer_indices {sel} out of range for {len(layers)} layers"
            )
    if L_sel != len(sel):
        raise ValueError(
            f"x_e has {L_sel} layer slices but {len(sel)} target layers "
            f"({'all' if layer_indices is None else 'layer_indices'})"
        )
    slice_of_layer = {l: i for i, l in enumerate(sel)}

    cos, sin = _prefix_position_embeddings(transformer, x_e, P_eff)

    cache = DynamicCache()
    for l, layer in enumerate(layers):
        attn = layer.self_attn
        head_dim = getattr(
            attn, "head_dim", x_e.shape[-1] // attn.config.num_attention_heads
        )
        if l in slice_of_layer:
            hs = x_e[:, slice_of_layer[l]]  # (B, P_eff, D)
            hidden_shape = (B, P_eff, -1, head_dim)
            # Mirror Qwen3Attention.forward exactly: norm is applied on the
            # (B, P, H_kv, head_dim) view BEFORE the (1,2) transpose.
            k = attn.k_proj(hs).view(hidden_shape)
            if hasattr(attn, "k_norm"):
                k = attn.k_norm(k)
            k = k.transpose(1, 2)  # (B, H_kv, P_eff, head_dim)
            v = attn.v_proj(hs).view(hidden_shape).transpose(1, 2)

            c = cos.to(dtype=k.dtype, device=k.device)
            s = sin.to(dtype=k.dtype, device=k.device)
            k = (k * c) + (_rotate_half(k) * s)
        else:
            # Unselected layer: empty prefix → attends to real tokens only.
            n_kv = attn.k_proj.out_features // head_dim
            k = x_e.new_zeros((B, n_kv, 0, head_dim))
            v = x_e.new_zeros((B, n_kv, 0, head_dim))

        cache.update(k, v, l)
    return cache
