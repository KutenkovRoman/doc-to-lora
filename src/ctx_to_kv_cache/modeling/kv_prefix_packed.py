from __future__ import annotations

import contextlib

import torch

from flash_attn import flash_attn_varlen_func
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ctx_to_kv_cache.modeling.kv_prefix import _get_layers


_ATTN_NAME = "kv_prefix"
# Per-process stack of active contexts (forward is sequential within a
# process; multi-GPU = separate processes). Lets the single registered
# attention fn find its segment metadata without threading it through
# every transformers call.
_CTX_STACK: list["_PackedKVContext"] = []


class _PackedKVContext:
    """
    Segment geometry shared by every layer's attention in one forward.

    ``P_d`` (prefix length) is fixed per document across all layers
    (``build_kv_prefix_cache`` uses one ``P_eff`` for the whole model), so
    the ``cu_seqlens`` are computed once here and reused for every layer.
    """

    def __init__(self, doc_caches, doc_of_seg, bounds, device):
        self.doc_caches = doc_caches        # list[DynamicCache], one per doc
        self.doc_of_seg = doc_of_seg        # list[int], len n_seg
        self.bounds = bounds                # list[int], len n_seg+1
        n_seg = len(doc_of_seg)
        seg_len = [bounds[s + 1] - bounds[s] for s in range(n_seg)]
        # Per-doc prefix length = max over layers (a layer-subset leaves
        # unselected layers at 0-length; selected layers all share P_eff for a
        # given doc), so the max recovers the doc's true prefix length without
        # assuming layer 0 is selected.
        def _p_eff(dc):
            return max(
                (dc.key_cache[l].shape[2] for l in range(len(dc.key_cache))),
                default=0,
            )
        p_eff = [_p_eff(doc_caches[doc_of_seg[s]]) for s in range(n_seg)]
        cu_q = [0]
        cu_k = [0]
        cu_k_np = [0]  # no-prefix variant for unselected layers
        for s in range(n_seg):
            cu_q.append(cu_q[-1] + seg_len[s])
            cu_k.append(cu_k[-1] + p_eff[s] + seg_len[s])
            cu_k_np.append(cu_k_np[-1] + seg_len[s])
        self.cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, device=device)
        self.cu_seqlens_k = torch.tensor(cu_k, dtype=torch.int32, device=device)
        self.cu_seqlens_k_noprefix = torch.tensor(
            cu_k_np, dtype=torch.int32, device=device
        )
        self.max_seqlen_q = max(seg_len) if seg_len else 0
        self.max_seqlen_k = (
            max(p_eff[s] + seg_len[s] for s in range(n_seg)) if n_seg else 0
        )
        self.max_seqlen_k_noprefix = max(seg_len) if seg_len else 0
        # Which absolute layers carry a prefix (global: the selected set is the
        # same for every doc). All-layers default → every layer, so the
        # attention below always takes the with-prefix branch (unchanged).
        dc0 = doc_caches[0]
        self.prefix_layers = {
            l for l in range(len(dc0.key_cache)) if dc0.key_cache[l].shape[2] > 0
        }


def _kv_prefix_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling=None,
    dropout=0.0,
    sliding_window=None,
    **kwargs,
):
    """
    ``ALL_ATTENTION_FUNCTIONS`` callable — prefix-LM varlen attention.

    Shapes follow the Qwen3Attention call site: ``query`` is
    ``(1, n_heads, tot_len, hd)``, ``key``/``value`` are
    ``(1, n_kv, tot_len, hd)`` (already q_norm/k_norm'd + RoPE'd at the
    shifted positions). Returns ``(attn_output, None)`` with
    ``attn_output`` shaped ``(1, tot_len, n_heads, hd)`` — exactly what
    Qwen3Attention then ``.reshape(*input_shape, -1)``-s.
    """
    if sliding_window is not None:
        raise NotImplementedError(
            "kv_prefix packed attention does not support sliding-window layers"
        )
    if not _CTX_STACK:
        raise RuntimeError(
            "_kv_prefix_attention invoked with empty _CTX_STACK — "
            "config._attn_implementation='kv_prefix' leaked outside "
            "kv_prefix_packed_attention()"
        )
    ctx = _CTX_STACK[-1]
    l = module.layer_idx
    # This layer's prefix is 0-length when it's not in the selected set
    # (build_kv_prefix_cache stored an empty entry), so the prepend below is a
    # no-op and we must use the no-prefix cu_seqlens for it.
    use_prefix = l in ctx.prefix_layers
    cu_k = ctx.cu_seqlens_k if use_prefix else ctx.cu_seqlens_k_noprefix
    max_k = ctx.max_seqlen_k if use_prefix else ctx.max_seqlen_k_noprefix

    q = query[0].transpose(0, 1).contiguous()   # (tot_len, n_heads, hd)
    k_real = key[0].transpose(0, 1)             # (tot_len, n_kv,  hd)
    v_real = value[0].transpose(0, 1)

    k_parts: list[torch.Tensor] = []
    v_parts: list[torch.Tensor] = []
    for s, d in enumerate(ctx.doc_of_seg):
        lo, hi = ctx.bounds[s], ctx.bounds[s + 1]
        ke = ctx.doc_caches[d].key_cache[l][0].transpose(0, 1)    # (P_d,n_kv,hd)
        ve = ctx.doc_caches[d].value_cache[l][0].transpose(0, 1)
        k_parts.append(ke.to(k_real.dtype))
        k_parts.append(k_real[lo:hi])
        v_parts.append(ve.to(v_real.dtype))
        v_parts.append(v_real[lo:hi])
    k_cat = torch.cat(k_parts, dim=0).contiguous()
    v_cat = torch.cat(v_parts, dim=0).contiguous()

    out = flash_attn_varlen_func(
        q,
        k_cat,
        v_cat,
        cu_seqlens_q=ctx.cu_seqlens_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=ctx.max_seqlen_q,
        max_seqlen_k=max_k,
        dropout_p=dropout if module.training else 0.0,
        softmax_scale=scaling,
        causal=True,
    )                                           # (tot_len, n_heads, hd)
    return out.unsqueeze(0), None


@contextlib.contextmanager
def kv_prefix_packed_attention(base_model, doc_caches, doc_of_seg, bounds, device):
    """
    Scope: route ``base_model``'s attention through the prefix-LM varlen fn.

    Restores ``config._attn_implementation``, ``Model._update_causal_mask``
    and the ``ALL_ATTENTION_FUNCTIONS`` entry on exit (a trusted, fully
    scoped swap — every mutation is undone in the ``finally``).
    """
    transformer, _ = _get_layers(base_model)
    ctx = _PackedKVContext(doc_caches, doc_of_seg, list(bounds), device)

    cfg = base_model.config
    old_impl = getattr(cfg, "_attn_implementation", None)
    old_mask = transformer._update_causal_mask
    had_entry = _ATTN_NAME in ALL_ATTENTION_FUNCTIONS
    prev_fn = ALL_ATTENTION_FUNCTIONS.get(_ATTN_NAME) if had_entry else None

    _CTX_STACK.append(ctx)
    try:
        ALL_ATTENTION_FUNCTIONS.update({_ATTN_NAME: _kv_prefix_attention})
        cfg._attn_implementation = _ATTN_NAME
        transformer._update_causal_mask = lambda *a, **k: None
        yield
    finally:
        _CTX_STACK.pop()
        cfg._attn_implementation = old_impl
        transformer._update_causal_mask = old_mask
        if had_entry:
            ALL_ATTENTION_FUNCTIONS.update({_ATTN_NAME: prev_fn})
        else:
            ALL_ATTENTION_FUNCTIONS.pop(_ATTN_NAME, None)
