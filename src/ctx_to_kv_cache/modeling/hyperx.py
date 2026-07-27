from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from math import sqrt
from typing import Any, Literal

import torch
from torch import Tensor, nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from einops.layers.torch import EinMix as Mix

import sys
sys.path.append("src")  # complains that there is no module ctx_to_kv_cache without this

from ctx_to_kv_cache.configs import CtxEncoderArguments
from ctx_to_kv_cache.data.ctx_chunking import context_to_padded_tensor
from ctx_to_kv_cache.model_loading import get_model, get_tokenizer
from ctx_to_kv_cache.modeling.aggregator import (
    AGGREGATOR_CLS,
    AGGREGATOR_TYPE,
    AggregatorConfig,
)
from ctx_to_kv_cache.modeling.ctx_encoder import CTX_ENCODER_CLS
from ctx_to_kv_cache.modeling.kv_prefix import _get_layers, build_kv_prefix_cache
from ctx_to_kv_cache.modeling.kv_prefix_packed import kv_prefix_packed_attention

from ctx_to_lora.modeling.hypernet import (
    AggregatorConfig,
    LoraConfig,
    HypernetConfig,
    PeftType,
    TaskType,
    LoraRuntimeConfig,
)
torch.serialization.add_safe_globals(
    [
        AggregatorConfig,
        LoraConfig,
        HypernetConfig,
        PeftType,
        TaskType,
        LoraRuntimeConfig,
        set,  # for real?
    ]
)

XPrefixMode = Literal["shallow", "kv"]


# ============================================================
# Config
# ============================================================


ChunkCombineMode = Literal["concat", "average", "sum"]


@dataclass
class HyperXConfig:
    """Architecture config for HyperX hypernet."""

    mode: XPrefixMode = "shallow"
    num_prefix_tokens: int = 20       # P
    hidden_size: int = 2560           # D (must match base model)
    num_layers: int = 36              # L (must match base model)
    # kv mode: which base-model layers get a K/V prefix. None = all `num_layers`
    # (default). A subset shrinks the hypernet output head (M = len·P) and skips
    # the prefix on unselected layers — lowering the memory floor. Absolute base
    # layer indices, 0-based. Ignored in shallow mode.
    layer_indices: tuple[int, ...] | None = None
    latent_size: int = 2048           # d_latent (Perceiver internal dim)
    init_scale: float = 0.02          # std for output head init
    use_out_norm: bool = True         # LayerNorm on X_p output

    # Context tokenization/chunking contract — MUST match what the training
    # dataset used, or the hypernet sees a different input distribution at
    # eval time than it was trained on. Serialized in the checkpoint so
    # `internalize()` reproduces the exact train-time chunking.
    max_ctx_chunk_len: int = 512
    max_ctx_chunks: int = 8

    # How to merge per-chunk prefixes when the doc is split into K chunks:
    #   - "concat" (default): concatenate by seq dim → final prefix length P·K
    #     (analog of HyperLoRA's stack-by-rank; max expressiveness).
    #   - "average": element-wise mean across chunks → final length P
    #     (analog of HyperLoRA's avg_chunk_loras=True; bounded cost).
    #   - "sum": element-wise sum across chunks → final length P
    #     (no precedent in HyperLoRA — exploratory mode).
    chunk_combine_mode: ChunkCombineMode = "concat"

    # Aggregator (Perceiver) settings — internally we set num_layers /
    # num_modules / lora_r so the Perceiver outputs M = num_latents queries.
    aggregator_config: AggregatorConfig | None = None

    @property
    def n_prefix_layers(self) -> int:
        """
        How many base layers receive a prefix (kv mode).

        len(layer_indices) if a subset is set, else all num_layers.
        """
        if self.layer_indices is not None:
            return len(self.layer_indices)
        return self.num_layers

    @property
    def num_latents(self) -> int:
        """
        Total number of Perceiver output queries M.

        shallow → P (one prefix). kv → (n_prefix_layers)·P (one P-row prefix
        per *selected* layer).
        """
        if self.mode == "shallow":
            return self.num_prefix_tokens
        return self.n_prefix_layers * self.num_prefix_tokens

    def make_aggregator_config(
        self, ctx_encoder_hidden: int, perceiver_args: dict[str, Any]
    ) -> AggregatorConfig:
        """
        Build AggregatorConfig that yields M output queries.

        We co-opt the (num_layers × num_modules × r) factor in the existing
        Perceiver to mean (L_eff × P × 1) where L_eff = 1 for shallow and
        L_eff = L for kv.
        """
        L_eff = 1 if self.mode == "shallow" else self.n_prefix_layers
        return AggregatorConfig(
            aggregator_type=AGGREGATOR_TYPE.PERCEIVER,
            feature_size=ctx_encoder_hidden,
            output_size=self.latent_size,
            num_layers=L_eff,
            num_modules=self.num_prefix_tokens,
            num_extra_modules=0,
            pooling_type="last_token",  # unused for perceiver
            num_latent_factor=1,        # unused
            lora_r=1,
            per_rank_gen=False,
            **perceiver_args,
        )


# ============================================================
# Hypernet
# ============================================================

# unused
class ResMLPBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout_rate: float = 0,
    ):
        super().__init__()

        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size),
        ]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.mlp(x)


class ResMLPBlockPerLayer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ):
        super().__init__()

        layers = [
            nn.LayerNorm(input_size),
            Mix(
                "bs n_layers n_prefix d_in -> bs n_layers n_prefix d_hid",
                weight_shape="n_layers d_in d_hid",
                bias_shape="n_layers d_hid",
                n_layers=n_layers,
                d_in=input_size,
                d_hid=hidden_size,
            ),
            nn.SiLU(),
            Mix(
                "bs n_layers n_prefix d_hid -> bs n_layers n_prefix d_out",
                weight_shape="n_layers d_hid d_out",
                bias_shape="n_layers d_out",
                n_layers=n_layers,
                d_hid=hidden_size,
                d_out=output_size,
            ),
            nn.LayerNorm(output_size),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.layers(x)


class HyperX(nn.Module):
    """
    Hypernet: document context → X-prefix tensor.

    Forward signature mirrors `HyperLoRA.forward`:
        features: (B, [L_ctx,] N, D_ctx) — output of ctx_encoder
    Returns:
        x_prefix: shape depends on mode
            shallow: (B, P, D)
            kv:      (B, L, P, D)
    """

    def __init__(self, config: HyperXConfig):
        super().__init__()
        self.config = config
        self.P = config.num_prefix_tokens
        self.D = config.hidden_size
        self.L = config.num_layers
        self.d_latent = config.latent_size
        # self.mode = config.mode  # no need for it

        agg_config = config.aggregator_config
        if agg_config is None:
            raise ValueError(
                "HyperXConfig.aggregator_config must be set; use "
                "HyperXConfig.make_aggregator_config(...) to construct it."
            )
        self.aggregator = AGGREGATOR_CLS[agg_config.aggregator_type](
            **vars(agg_config)
        )

        # Output head: project d_latent → D
        # --------------------------------------------------------------- #
        self.head = nn.Linear(self.d_latent, self.D)
        nn.init.normal_(
            self.head.weight,
            mean=0.0,
            std=(config.init_scale / sqrt(self.d_latent)),
        )
        nn.init.zeros_(self.head.bias)
        # --------------------------------------------------------------- #
        # self.prehead = ResMLPBlockPerLayer(
        #     n_layers=config.n_prefix_layers,
        #     input_size=self.d_latent,
        #     hidden_size=self.d_latent * 4,
        #     output_size=self.d_latent,
        # )
        # self.head = Mix(
        #     "bs n_layers n_prefix d_latent -> bs n_layers n_prefix d_kv",
        #     weight_shape="n_layers d_latent d_kv",
        #     bias_shape=None,  # no bias
        #     n_layers=config.n_prefix_layers,
        #     d_latent=self.d_latent,
        #     n_prefix=self.P,
        #     d_kv=self.D,
        # )
        # --------------------------------------------------------------- #

        if config.use_out_norm:
            self.out_norm = nn.LayerNorm(self.D)
            # LayerNorm normalises to unit variance *before* applying its
            # gain, so the tiny head init does NOT keep the prefix small once
            # LN is present — the output magnitude is set purely by the LN
            # gain. Init the gain to `init_scale` so the prefix std is approx
            # init_scale at step 0 (small but non-zero → near the frozen-base
            # behaviour, stable warm start) while keeping a learnable
            # per-feature scale. (The small head init still matters when
            # use_out_norm=False.)
            nn.init.constant_(self.out_norm.weight, config.init_scale)
            nn.init.zeros_(self.out_norm.bias)
        else:
            self.out_norm = nn.Identity()

    def forward(
        self,
        features: Tensor,
        attn_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        # Mirror HyperLoRA.forward (hypernet.py:401): the hypernet body runs
        # under bf16 autocast on CUDA so it matches the bf16 ctx_encoder
        # features and the bf16 base model. Guarded on CUDA so CPU dummy
        # tests (fp32, no autocast) still run. This is the canonical D2L
        # precision pattern — an earlier review removed it, which broke the
        # plain-python (non-accelerate-launch) path with a dtype mismatch.
        if features.is_cuda:
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            amp_ctx = contextlib.nullcontext()

        with amp_ctx:
            # (B, L_eff, P, d_latent)
            agg_out, _ = self.aggregator(features, attn_mask, position_ids)

            # (B, L_eff, P, D)
            # --------------------------------------------------------------- #
            x = self.head(agg_out)
            # --------------------------------------------------------------- #
            # x_emb = self.prehead(agg_out)
            # norm = torch.norm(x_emb, dim=-1, keepdim=True)
            # norm_x_emb = x_emb / norm
            # x = self.head(norm_x_emb)
            # --------------------------------------------------------------- #

            if self.config.use_out_norm:
                x = self.out_norm(x)

        # kv: keep (B, L, P, D) — one P-row virtual KV source per layer
        return x


# ============================================================
# Wrapper (analog of ModulatedPretrainedModel for HyperLoRA)
# ============================================================


class ModulatedPretrainedModelX(nn.Module):
    """
    Base LLM + ctx_encoder + HyperX hypernet

    Mirrors `ModulatedPretrainedModel` from `hypernet.py` for the LoRA path.
    """

    def __init__(
        self,
        base_model: nn.Module,
        hyperx_config: HyperXConfig,
        ctx_encoder_args: CtxEncoderArguments,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.hyperx_config = hyperx_config
        self.ctx_encoder_args = ctx_encoder_args
        self.device = next(base_model.parameters()).device

        # Build ctx_encoder (frozen Qwen3-4B as default)
        ctx_model_name = ctx_encoder_args.ctx_encoder_model_name_or_path
        if ctx_model_name is None:
            ctx_model_name = base_model.config.name_or_path
        base_model_attn_impl = base_model.config._attn_implementation
        encoder_model = get_model(
            ctx_model_name,
            train=False,
            requires_grad=False,
            use_flash_attn=(base_model_attn_impl == "flash_attention_2"),
            use_q_lora=ctx_encoder_args.quantize_ctx_encoder,
        )
        self.ctx_encoder = CTX_ENCODER_CLS[ctx_encoder_args.ctx_encoder_type](
            encoder_model, ctx_encoder_args
        )

        # Build HyperX hypernet
        self.hyperx = HyperX(hyperx_config).to(self.device)

        # Cached prefix artifact from the last internalize():
        #   kv:      (1, L, P_eff, D) — per-call source for the KV-prefix cache
        self._infer_x_prefix: Tensor | None = None

    def train(self, mode: bool = True):
        """Keep only HyperX trainable when the wrapper enters train mode."""
        super().train(mode)
        if mode:
            self.base_model.eval()
            self.ctx_encoder.eval()
            self.hyperx.train()
        return self

    def load_d2l_aggregator(self):
        ckpt = torch.load("/home/jovyan/veprikov/Doc-To-Lora/doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin", weights_only=True)

        aggregator_state_dict = {
            k.removeprefix("aggregator."): v for k, v in ckpt.items() if k.startswith("aggregator.")
        }
        self.hyperx.aggregator.load_state_dict(aggregator_state_dict)

        for param in self.hyperx.aggregator.parameters():
            param.requires_grad = False
        self.hyperx.aggregator.eval()

    # ---- core forward ----

    def _xprefix_per_chunk(
        self,
        ctx_ids: Tensor,
        ctx_attn_mask: Tensor | None = None,
        ctx_position_ids: Tensor | None = None,
    ) -> Tensor:
        """ctx_encoder + hypernet → PER-CHUNK X-prefix (not yet combined).

        Returns the hypernet output before chunk-combine:
          shallow: `(n_chunks, P, D)`   kv: `(n_chunks, L, P, D)`

        `n_chunks` is the aggregator batch dim. It works for two callers:
          - single-doc (SFT / `internalize`): `ctx_ids` is
            `(n_chunks, chunk_len)`, all chunks of one document.
          - packed pretrain: `ctx_ids` is `(1, total_ctx_tokens)` flat with
            chunk boundaries in `ctx_position_ids` (resets to 0 per chunk) —
            the ctx_encoder + Perceiver delimit chunks by `position_ids`
            exactly like HyperLoRA's `generate_weights`, so the aggregator
            still emits one row per chunk. Per-doc grouping (by
            `n_ctx_chunks`) is the caller's job (see
            `_forward_packed_kv_train`).
        """
        # The Perceiver aggregator requires either an attention mask or
        # position ids. At inference `internalize()` has neither (it just
        # tokenizes the doc), so default the mask to all-ones — same fallback
        # HyperLoRA uses (hypernet.py: `ctx_attn_mask = torch.ones_like(ctx_ids)`).
        if ctx_attn_mask is None and ctx_position_ids is None:
            ctx_attn_mask = torch.ones_like(ctx_ids)

        mb = int(os.environ.get("HYPERX_CHUNK_MICROBATCH", "0"))

        # Packed-pretrain micro-batch: ctx_ids is a single flat
        # (1, total_ctx_tokens) stream with chunk boundaries in
        # ctx_position_ids resets (see `_forward_packed_kv_train`). The dim-0
        # micro-batch below would short-circuit here (ctx_features.shape[0]==1),
        # so we split at chunk boundaries explicitly: process `mb` chunks of
        # ctx_ids at a time through ctx_encoder + Perceiver, concat outputs.
        # Mathematically identical to the single-call forward modulo bf16
        # reduction order (chunks are independent — paper §3.2; per-sub-stream
        # cu_seqlens are exactly the corresponding slice of the full-stream
        # cu_seqlens, and `self.latents_q` carries no cross-chunk state).
        # Cuts perceiver `bsz_aggr` from `L × total_chunks` to `L × mb`,
        # dominated by context RMSNorm fp32 buffers (~600 MB/block at L=36,
        # total_ctx_tokens=2048). Default off → byte-identical legacy path.
        if (
            mb > 0
            and ctx_ids.shape[0] == 1
            and ctx_position_ids is not None
            and ctx_attn_mask is None
        ):
            pid_row = ctx_position_ids[0]
            starts = (pid_row == 0).nonzero(as_tuple=True)[0]
            total_chunks = int(starts.numel())
            if total_chunks > mb:
                bounds = torch.cat([
                    starts,
                    torch.tensor(
                        [pid_row.numel()],
                        device=starts.device,
                        dtype=starts.dtype,
                    ),
                ]).tolist()
                outs = []
                for g0 in range(0, total_chunks, mb):
                    g1 = min(g0 + mb, total_chunks)
                    sub_ids = ctx_ids[:, bounds[g0]:bounds[g1]]
                    sub_pids = ctx_position_ids[:, bounds[g0]:bounds[g1]]
                    with torch.no_grad():
                        sub_feats = self.ctx_encoder(
                            input_ids=sub_ids,
                            attention_mask=None,
                            position_ids=sub_pids,
                        )
                    if self.hyperx_config.mode == "shallow":
                        sub_feats = sub_feats[:, -1:, :, :]
                    elif self.hyperx_config.layer_indices is not None:
                        sel = list(self.hyperx_config.layer_indices)
                        sub_feats = sub_feats[:, sel, :, :]
                    outs.append(self.hyperx(sub_feats, None, sub_pids))
                return torch.cat(outs, dim=0)

        with torch.no_grad():
            ctx_features = self.ctx_encoder(
                input_ids=ctx_ids,
                attention_mask=ctx_attn_mask,
                position_ids=ctx_position_ids,
            )
        # ctx_features: (n_chunks, n_enc_layers, seq, D) from
        # per_layer_activations (n_enc_layers = 36 for Qwen3-4B with
        # ctx_encoder_last_layer=None). The Perceiver aggregator runs in
        # layer_to_layer mode, which flattens (bs, n_layers, seq, D) →
        # (n_layers·bs, seq, D) and repeats the attn mask by aggregator
        # num_layers. That only stays batch-consistent if n_enc_layers
        # equals the aggregator's num_layers:
        #   - kv:      aggregator num_layers = L = n_enc_layers → keep all
        #   - shallow: aggregator num_layers = 1 → collapse to the final
        #     layer (the standard single-document representation) so the
        #     flatten matches the [n_chunks, seq] attention mask.
        if self.hyperx_config.mode == "shallow":
            ctx_features = ctx_features[:, -1:, :, :]  # (n_chunks, 1, seq, D)
        elif self.hyperx_config.layer_indices is not None:
            # kv layer subset: keep only the selected base layers' encoder
            # features (encoder layer l ↔ base layer l) so the aggregator
            # (num_layers = len(indices)) emits a P-row prefix ONLY for those
            # layers. Mirrors the shallow slice above; n_enc_layers must stay
            # == aggregator num_layers for the layer_to_layer fold to be
            # batch-consistent (see the keep-all note below).
            sel = list(self.hyperx_config.layer_indices)
            ctx_features = ctx_features[:, sel, :, :]  # (n_chunks, len(sel), seq, D)
        # hypernet returns (n_chunks, [L,] P, D) — treats n_chunks as batch.
        # SFT/eval micro-batch path (ctx_ids = (n_chunks, chunk_len)): the
        # packed-pretrain variant above handles ctx_ids = (1, total_ctx_tokens)
        # by splitting at chunk boundaries. Long-context eval (e.g., NIAH at
        # 24K+ tokens) OOMs at the all-chunks-at-once forward; same env var
        # `HYPERX_CHUNK_MICROBATCH` covers both paths.
        n_chunks = ctx_features.shape[0]
        if mb <= 0 or n_chunks <= mb:
            return self.hyperx(ctx_features, ctx_attn_mask, ctx_position_ids)
        outs = []
        for i in range(0, n_chunks, mb):
            sl = slice(i, i + mb)
            am = ctx_attn_mask[sl] if ctx_attn_mask is not None else None
            pid = ctx_position_ids[sl] if ctx_position_ids is not None else None
            outs.append(self.hyperx(ctx_features[sl], am, pid))
        return torch.cat(outs, dim=0)

    def generate_xprefix(
        self,
        ctx_ids: Tensor,
        ctx_attn_mask: Tensor | None = None,
        ctx_position_ids: Tensor | None = None,
    ) -> Tensor:
        """Single-document X-prefix: per-chunk → combined.

        Input `ctx_ids`: `(n_chunks, chunk_len)` (all chunks of ONE doc).
        Returns:
          - shallow + concat:  `(1, P · n_chunks, D)`
          - shallow + avg/sum: `(1, P, D)`
          - kv + concat:       `(1, L, P · n_chunks, D)`
          - kv + avg/sum:      `(1, L, P, D)`
        """
        x_prefix = self._xprefix_per_chunk(
            ctx_ids, ctx_attn_mask, ctx_position_ids
        )
        return self._combine_chunk_prefixes(x_prefix)

    def _combine_chunk_prefixes(self, x_prefix: Tensor) -> Tensor:
        """Merge `n_chunks` per-chunk prefixes into one final prefix.

        x_prefix shape:
          - shallow: `(n_chunks, P, D)` → out `(1, P_eff, D)`
          - kv:      `(n_chunks, L, P, D)` → out `(1, L, P_eff, D)`
        where `P_eff = P · n_chunks` (concat) or `P` (average/sum).
        """
        mode = self.hyperx_config.chunk_combine_mode
        if self.hyperx_config.mode == "shallow":
            # x_prefix: (n_chunks, P, D)
            if mode == "concat":
                # → (1, n_chunks · P, D)
                return x_prefix.reshape(1, -1, x_prefix.shape[-1])
            if mode == "average":
                return x_prefix.mean(dim=0, keepdim=True)  # (1, P, D)
            if mode == "sum":
                return x_prefix.sum(dim=0, keepdim=True)  # (1, P, D)
        else:  # kv
            # x_prefix: (n_chunks, L, P, D)
            if mode == "concat":
                # → (1, L, n_chunks · P, D)
                n, L, P, D = x_prefix.shape
                return x_prefix.permute(1, 0, 2, 3).reshape(1, L, n * P, D)
            if mode == "average":
                return x_prefix.mean(dim=0, keepdim=True)  # (1, L, P, D)
            if mode == "sum":
                return x_prefix.sum(dim=0, keepdim=True)  # (1, L, P, D)
        raise ValueError(f"Unknown chunk_combine_mode: {mode}")

    def forward(
        self,
        *args,
        ctx_ids: Tensor | None = None,
        ctx_attn_mask: Tensor | None = None,
        ctx_position_ids: Tensor | None = None,
        n_ctx_chunks: Tensor | None = None,
        n_queries: Tensor | None = None,
        return_generated_lora: bool = False,
        input_ids: Tensor | None = None,
        labels: Tensor | None = None,
        **kwargs,
    ):
        """Training forward.

        Computes the X artifact from ctx_ids, then runs the base model.

        kv: per-layer X_e becomes a pre-filled `past_key_values` cache. The
            real tokens are the ONLY queries, so logits are (B, N, V) and
            already aligned with the N input tokens — labels need NO prefix
            padding. The cache is differentiable w.r.t. the hypernet.

        Packed pretrain path (`train.py` + `DistillationTrainer`): when
        `n_ctx_chunks` is supplied the batch is a single packed sequence of
        many (ctx, prompt, response) segments (the canonical D2L self_gen
        pipeline). HyperLoRA handles this by per-segment Linear patching over
        an FA2-varlen forward; HyperX-kv's mechanism is *extra K/V rows*,
        which is a KV-cache concept with no clean seam in the cache-less
        varlen path. So `kv` processes the packed batch per segment through
        the verified `build_kv_prefix_cache` forward and re-assembles logits
        at the original packed positions — the KL loss code is unchanged and
        sees a `(1, tot_len, V)` logits tensor exactly like HyperLoRA. The
        offline teacher signal (`logprobs_vals/indices`) is model-agnostic.
        """
        if n_ctx_chunks is not None:
            outputs = self._forward_packed_kv_train(
                ctx_ids=ctx_ids,
                ctx_attn_mask=ctx_attn_mask,
                ctx_position_ids=ctx_position_ids,
                n_ctx_chunks=n_ctx_chunks,
                n_queries=n_queries,
                input_ids=input_ids,
                position_ids=kwargs.get("position_ids"),
            )
            return (outputs, (None, None)) if return_generated_lora else outputs

        if ctx_ids is None:
            return self.base_model(
                input_ids=input_ids, labels=labels, **kwargs
            )

        x_prefix = self.generate_xprefix(ctx_ids, ctx_attn_mask)

        # x_prefix: (1, L, P_eff, D). Broadcast to the batch and turn it
        # into per-layer virtual K/V via the model's own W_k/W_v/RoPE.
        B = input_ids.shape[0]
        x_e = (
            x_prefix.expand(B, -1, -1, -1)
            if x_prefix.shape[0] == 1 and B > 1
            else x_prefix
        )
        past = build_kv_prefix_cache(self.base_model, x_e)
        p_eff = x_e.shape[2]
        attn_mask = kwargs.pop("attention_mask", None)
        if attn_mask is None:
            attn_mask = torch.ones_like(input_ids)
        prefix_mask = torch.ones(
            (attn_mask.shape[0], p_eff),
            dtype=attn_mask.dtype,
            device=attn_mask.device,
        )
        full_mask = torch.cat([prefix_mask, attn_mask], dim=1)
        return self.base_model(
            input_ids=input_ids,
            attention_mask=full_mask,
            labels=labels,
            past_key_values=past,
            use_cache=True,
            **kwargs,
        )

    # ---- inference API (matches ModulatedPretrainedModel for HyperLoRA) ----

    @torch.no_grad()
    def internalize(self, ctx_str: str) -> None:
        """Inference: read document, set prefix. Call before .generate().

        The context is tokenized with the EXACT same chat-template +
        chunking + per-chunk-affix + chunk-cap logic the training dataset
        uses (shared `ctx_chunking` helper, parameterised by the chunk
        settings serialized in `HyperXConfig`). This is a correctness
        requirement: a raw `tokenizer(doc)` here would feed the hypernet a
        completely different input distribution than it saw at train time.

        kv: nothing is installed on the model here; the per-layer X_e is
        cached and `generate()` builds a fresh `past_key_values` from it on
        every call (robust to repeated generate() after one internalize()).
        """
        ctx_model_name = (
            getattr(self.ctx_encoder.base_model, "name_or_path", None) or
            self.ctx_encoder.base_model.config.name_or_path
        )
        ctx_tokenizer = get_tokenizer(ctx_model_name)
        ctx_ids, ctx_attn_mask = context_to_padded_tensor(
            ctx_tokenizer,
            ctx_str,
            max_ctx_chunk_len=self.hyperx_config.max_ctx_chunk_len,
            max_ctx_chunks=self.hyperx_config.max_ctx_chunks,
            device=self.device,
        )
        x_prefix = self.generate_xprefix(ctx_ids, ctx_attn_mask)
        self._infer_x_prefix = x_prefix

    def reset(self) -> None:
        """Inference: clear prefix."""
        self._infer_x_prefix = None

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        ctx_ids: Tensor | None = None,
        ctx_attn_mask: Tensor | None = None,
        ctx_position_ids: Tensor | None = None,
        n_ctx_chunks: Tensor | None = None,
        **gen_kwargs,
    ):
        """
        Eval-path entry: build the X-prefix INLINE from the collator's
        already-chunked ``ctx_ids`` (mirrors ``ModulatedPretrainedModel.
        generate``), then delegate to the validated prefix-driven body.

        run_eval/Seq2SeqTrainer chunks context with the correct uncapped
        1024-token ``split_too_long_ctx`` (the paper NIAH recipe) and passes
        the chunks as ``ctx_ids``/``n_ctx_chunks`` in the batch. We must NOT
        route through ``internalize()`` here — that re-chunks via
        ``ctx_chunking`` capped at ``hyperx_config.max_ctx_chunks`` (=8),
        silently truncating any context >~4 K tokens and dropping the
        needle on every long NIAH bin.

        When ``ctx_ids`` is None the prefix is taken from a prior
        ``internalize()`` (zoo path) — unchanged.
        """
        if ctx_ids is None:
            return self._generate_with_prefix(
                input_ids, attention_mask, **gen_kwargs
            )
        # generation_collator emits one n_ctx_chunks entry per document; a
        # batched prefix would apply doc-0's needle to the whole batch
        # (silent wrong answers). Fail loud — HyperX NIAH eval is bs=1.
        if (
            n_ctx_chunks is not None
            and int(torch.as_tensor(n_ctx_chunks).reshape(-1).numel()) > 1
        ):
            raise ValueError(
                "HyperX eval requires eval_batch_size_gen=1 (one document "
                f"per generate call); got n_ctx_chunks="
                f"{torch.as_tensor(n_ctx_chunks).reshape(-1).tolist()} — a "
                "batched X-prefix would apply doc-0's context to the whole "
                "batch and silently corrupt results."
            )
        self._infer_x_prefix = self.generate_xprefix(
            ctx_ids, ctx_attn_mask, ctx_position_ids
        )
        try:
            return self._generate_with_prefix(
                input_ids, attention_mask, **gen_kwargs
            )
        finally:
            # Each eval sample must rebuild from its own ctx_ids; don't
            # leak this doc's prefix into the next generate() call.
            self._infer_x_prefix = None

    def _generate_with_prefix(
        self, input_ids: Tensor, attention_mask: Tensor | None = None, **gen_kwargs
    ):
        """Generate a completion using the prefix from the last internalize().

        shallow: builds `inputs_embeds = [X_p ; embed(input_ids)]` + a
        matching mask and calls the base model's `generate` (HF derives
        positions/RoPE/KV-cache for `P_eff + N` natively; the embed patch is
        bypassed). HF returns only new tokens with `inputs_embeds`, so we
        re-prepend `input_ids` for the conventional `[prompt ; completion]`
        layout. Batched / `num_return_sequences` / dict-output handled.

        kv: a self-contained decode loop (`_kv_generate`). HF's `generate`
        cannot be used directly here — `_get_initial_cache_position` assumes
        a non-empty `past_key_values` is a *prefix of `input_ids`* and slices
        the prompt by the cache length, which is wrong for a virtual KV
        prefix. The manual loop drives `base_model.forward` with an explicit
        `cache_position` so real tokens sit at positions `P_eff..` (RoPE
        applied natively by the model) — identical to the training forward.
        Returns `[prompt ; completion]` token ids.
        """
        if self._infer_x_prefix is None:
            return self.base_model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
            )

        return self._kv_generate(input_ids, attention_mask, **gen_kwargs)

    # ---- kv-mode decode loop ----

    @staticmethod
    def _pick_token(
        logits: Tensor,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Tensor:
        """Next-token selection: greedy or temperature/top-k/top-p sampling.

        `logits`: (B, V). Returns (B,) token ids. Matches HF's warper order
        (temperature → top-k → top-p → softmax → multinomial).
        """
        if not do_sample or temperature == 0:
            return logits.argmax(dim=-1)
        logits = logits.float() / max(temperature, 1e-5)
        if top_k and top_k > 0:
            k = min(top_k, logits.shape[-1])
            kth = logits.topk(k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove = probs - sorted_logits.softmax(dim=-1) >= top_p
            remove[..., 0] = False  # always keep the top token
            remove = remove.scatter(1, sorted_idx, remove)
            logits = logits.masked_fill(remove, float("-inf"))
        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def _kv_generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        **gen_kwargs,
    ) -> Tensor:
        """Autoregressive decode with a virtual KV prefix.

        Builds a fresh per-call `past_key_values` (length `P_eff`) from the
        internalized per-layer `X_e`, then runs `base_model.forward` directly
        with an explicit `cache_position` so real tokens occupy positions
        `P_eff .. P_eff+N-1` and the cache appends generated K/V after them —
        byte-for-byte the same positional/attention layout as training.
        """
        base = self.base_model
        x_e = self._infer_x_prefix
        B, N = input_ids.shape
        device = input_ids.device
        if x_e.shape[0] == 1 and B > 1:
            x_e = x_e.expand(B, -1, -1, -1)
        past = build_kv_prefix_cache(base, x_e)
        p_eff = x_e.shape[2]

        if attention_mask is None:
            attention_mask = torch.ones(
                (B, N), dtype=torch.long, device=device
            )
        prefix_mask = torch.ones(
            (B, p_eff), dtype=attention_mask.dtype, device=device
        )
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        max_new = gen_kwargs.get("max_new_tokens") or 512
        do_sample = bool(gen_kwargs.get("do_sample", False))
        temperature = float(gen_kwargs.get("temperature", 1.0) or 1.0)
        top_p = float(gen_kwargs.get("top_p", 1.0) or 1.0)
        top_k = int(gen_kwargs.get("top_k", 0) or 0)

        gcfg = base.generation_config
        eos = gen_kwargs.get("eos_token_id", gcfg.eos_token_id)
        if eos is None:
            eos_ids: list[int] = []
        else:
            if isinstance(eos, torch.Tensor):
                eos = eos.tolist()
            if isinstance(eos, (list, tuple)):
                eos_ids = [int(e) for e in eos]
            else:
                eos_ids = [int(eos)]
        pad = gen_kwargs.get("pad_token_id", gcfg.pad_token_id)
        if pad is None:
            pad = eos_ids[0] if eos_ids else 0

        cache_position = torch.arange(p_eff, p_eff + N, device=device)
        out = base(
            input_ids=input_ids,
            attention_mask=full_mask,
            past_key_values=past,
            use_cache=True,
            cache_position=cache_position,
        )
        logits = out.logits[:, -1, :]

        generated = input_ids
        unfinished = torch.ones(B, dtype=torch.bool, device=device)
        cur = p_eff + N
        for _ in range(int(max_new)):
            nxt = self._pick_token(logits, do_sample, temperature, top_p, top_k)
            nxt = torch.where(unfinished, nxt, nxt.new_full((), pad))
            generated = torch.cat([generated, nxt[:, None]], dim=1)
            for e in eos_ids:
                unfinished = unfinished & (nxt != e)
            if not bool(unfinished.any()):
                break
            # New column = 1 only for sequences still generating; finished
            # rows get 0 so their emitted pad tokens are not attended (matches
            # HF batched generation; the all-ones version let finished rows
            # attend to pad K/V).
            new_col = unfinished.to(full_mask.dtype).unsqueeze(1)
            full_mask = torch.cat([full_mask, new_col], dim=1)
            out = base(
                input_ids=nxt[:, None],
                attention_mask=full_mask,
                past_key_values=past,
                use_cache=True,
                cache_position=torch.tensor([cur], device=device),
            )
            logits = out.logits[:, -1, :]
            cur += 1
        return generated

    # ---- packed pretrain path (train.py + DistillationTrainer) ----

    def _forward_packed_kv_train(
        self,
        ctx_ids: Tensor,
        ctx_attn_mask: Tensor | None,
        ctx_position_ids: Tensor | None,
        n_ctx_chunks: Tensor,
        n_queries: Tensor | None,
        input_ids: Tensor,
        position_ids: Tensor | None,
    ):
        """Per-segment kv forward over one packed pretrain batch.

        Inputs (collator `flatten_if_not_packed` packed path, built by
        `pack_data_points_FA`):
          ctx_ids:          (1, total_ctx_tokens) — FLAT stream of every
                            chunk of every doc in the pack, concatenated
          ctx_position_ids: (1, total_ctx_tokens) — resets to 0 at each
                            chunk start (the only chunk delimiter)
          n_ctx_chunks:     (n_doc,)  chunks per document
          n_queries:        (n_doc,)  packed text segments per document (≥1)
          input_ids:        (1, tot_len)  packed real tokens, segments concat
          position_ids:     (1, tot_len)  resets to 0 at each segment start

        Returns a `CausalLMOutputWithPast` whose `.logits` is
        `(1, tot_len, V)` — base-model next-token logits over the real packed
        tokens, exactly the layout HyperLoRA produces, so `DistillationTrainer`
        (which indexes `logits[label_pos[0], label_pos[1]-1]`) is unchanged.
        Virtual prefix rows are K/V only (never queries) so they contribute
        no logits and need no position bookkeeping in the output.
        """
        base = self.base_model
        device = input_ids.device
        n_chunks = n_ctx_chunks.reshape(-1)
        n_chunks_list = n_chunks.tolist()
        if n_queries is None:
            seg_per_doc = [1] * len(n_chunks_list)
        else:
            seg_per_doc = n_queries.reshape(-1).tolist()

        # ctx_ids is the FLAT packed ctx stream (1, total_ctx_tokens); chunk
        # boundaries live in ctx_position_ids. Mirror HyperLoRA's
        # generate_weights EXACTLY: pass ctx_attn_mask THROUGH (it is None for
        # the packed stream — the collator emits no ctx_attn_mask) so the
        # Perceiver takes only the ctx_position_ids varlen branch and emits
        # ONE row per chunk → (total_chunks, [L,] P, D). Forcing the mask
        # non-None would make aggregator.py's two non-elif `if` blocks BOTH
        # fire (mask branch rearranges 4-D→3-D, position_ids branch then
        # assumes 4-D) → crash + train != HyperLoRA ctx distribution. The
        # both-None internalize fallback lives in _xprefix_per_chunk.
        per_chunk = self._xprefix_per_chunk(
            ctx_ids, ctx_attn_mask, ctx_position_ids
        )  # (total_chunks, L, P, D) for kv
        total_chunks = int(sum(n_chunks_list))
        if per_chunk.shape[0] != total_chunks:
            raise ValueError(
                f"hypernet emitted {per_chunk.shape[0]} chunk rows but "
                f"sum(n_ctx_chunks)={total_chunks}; ctx_position_ids chunk "
                f"delimiting does not match n_ctx_chunks"
            )
        groups = per_chunk.split(n_chunks_list, dim=0)  # one per doc
        x_e_per_doc = [
            self._combine_chunk_prefixes(g) for g in groups
        ]  # each (1, L, P_eff, D)

        # Segment boundaries from packed position_ids resets.
        pos = (position_ids if position_ids is not None
               else torch.arange(input_ids.shape[1], device=device)).reshape(-1)
        starts = (pos == 0).nonzero(as_tuple=True)[0]
        bounds = torch.cat([
            starts,
            torch.tensor([pos.numel()], device=starts.device),
        ]).tolist()
        n_seg = len(bounds) - 1
        n_seg_expected = int(sum(seg_per_doc))
        if n_seg != n_seg_expected:
            raise ValueError(
                f"packed segment count {n_seg} != sum(n_queries) "
                f"{n_seg_expected}; position_ids resets do not match the "
                f"collator contract"
            )
        # Map each packed segment → its document index.
        doc_of_seg = []
        for d, q in enumerate(seg_per_doc):
            doc_of_seg.extend([d] * int(q))

        ids_row = input_ids[0]

        # Two equivalent forwards over the pack:
        #   packed (DEFAULT): ONE varlen prefix-LM forward — the HyperLoRA-
        #     parity path. VALIDATED CORRECT: per-call fp32-equidistant vs
        #     serial at the kernel level + end-to-end loss tracks serial
        #     within run-to-run nondeterminism. Multi-GPU loss A/B
        #     (2026-05-28, 2×A100 DDP, 80 steps, NIAH-CE recipe): serial vs
        #     packed mean|Δloss|=0.016 (median 0.005), final 1.8534 vs 1.8560,
        #     and ~1.99× faster (890 s vs 447 s) → packed is the default.
        #   serial (opt-out `HYPERX_KV_SERIAL=1`): one `base.forward` per
        #     segment (batch=1) — the reference oracle, kept for debugging.
        # See plans/hyperx_kv_batched_packed_forward.md § VARLEN PREFIX-LM.
        if os.environ.get("HYPERX_KV_SERIAL") == "1":
            seg_logits = self._kv_seg_logits_serial(
                base, ids_row, bounds, x_e_per_doc, doc_of_seg, n_seg, device
            )
        else:
            seg_logits = self._kv_seg_logits_packed(
                base, ids_row, bounds, x_e_per_doc, doc_of_seg, n_seg, device
            )

        logits = torch.cat(seg_logits, dim=0).unsqueeze(0)  # (1, tot_len, V)
        if logits.shape[1] != input_ids.shape[1]:
            raise ValueError(
                f"reassembled logits len {logits.shape[1]} != packed "
                f"tot_len {input_ids.shape[1]} — segments did not tile the "
                f"packed sequence"
            )
        return CausalLMOutputWithPast(logits=logits)

    def _kv_seg_logits_serial(
        self, base, ids_row, bounds, x_e_per_doc, doc_of_seg, n_seg, device
    ):
        """Reference path: one base.forward per segment (batch=1, serial).

        The correctness oracle for the packed varlen forward, and the
        current default training path. Returns a list of `(seg_len_s, V)`
        logit tensors in packed order.
        """
        li = self.hyperx_config.layer_indices
        seg_logits = []
        for s in range(n_seg):
            lo, hi = bounds[s], bounds[s + 1]
            seg_ids = ids_row[lo:hi].unsqueeze(0)  # (1, len_s)
            x_e = x_e_per_doc[doc_of_seg[s]]
            p_eff = x_e.shape[2]
            seg_len = hi - lo
            if li is None:
                # All layers: vanilla HF forward over a prefilled cache — the
                # independent oracle (one uniform causal mask is correct since
                # every layer shares P_eff).
                past = build_kv_prefix_cache(base, x_e)
                cache_position = torch.arange(
                    p_eff, p_eff + seg_len, device=device
                )
                full_mask = torch.ones(
                    (1, p_eff + seg_len), dtype=torch.long, device=device
                )
                out = base(
                    input_ids=seg_ids,
                    attention_mask=full_mask,
                    past_key_values=past,
                    use_cache=True,
                    cache_position=cache_position,
                )
            else:
                # Layer subset: vanilla HF builds ONE causal mask for all
                # layers (uniform prefix length) → wrong when only some layers
                # carry a prefix. Route this single segment through the same
                # prefix-aware varlen attention as the packed path. Still
                # batch=1 / one forward per segment, but it shares attention
                # code with packed here, so it is not a fully independent
                # oracle for subset configs.
                cache = build_kv_prefix_cache(base, x_e, li)
                position_ids = torch.arange(
                    p_eff, p_eff + seg_len, device=device
                ).unsqueeze(0)
                with kv_prefix_packed_attention(
                    base, [cache], [0], [0, seg_len], device
                ):
                    out = base(
                        input_ids=seg_ids,
                        position_ids=position_ids,
                        use_cache=False,
                    )
            seg_logits.append(out.logits[0])  # (len_s, V)
        return seg_logits

    def _kv_seg_logits_packed(
        self, base, ids_row, bounds, x_e_per_doc, doc_of_seg, n_seg, device
    ):
        """Varlen prefix-LM: ONE forward over the whole pack.

        The HyperLoRA-parity path. Each segment's queries attend to its
        document's virtual K/V prefix + its own causal real tokens via
        `flash_attn_varlen_func` with distinct `cu_seqlens_q/k` (bottom-
        right causal == serial semantics). Mathematically identical to
        `_kv_seg_logits_serial` (same RoPE, same visibility); both use
        flash → expect only reduction-order drift. Returns the same list
        of `(seg_len_s, V)` logit tensors in packed order.
        """
        # Fail loud if base-model gradient checkpointing is on. The serial
        # path is implicitly protected (use_cache=True forces GC off in
        # Qwen3Model); this path uses use_cache=False, removing that guard.
        # Under base GC, the layer forward is recomputed in backward —
        # OUTSIDE the kv_prefix_packed_attention CM scope (popped by then)
        # → wrong recomputed attn → silently corrupted grads. Code-review
        # Finding 14.
        _tr, _ = _get_layers(base)
        if getattr(_tr, "gradient_checkpointing", False):
            raise RuntimeError(
                "HYPERX_KV_PACKED requires base-model gradient checkpointing "
                "OFF (backward recomputation would run outside the "
                "kv_prefix attention scope and corrupt gradients). Disable "
                "base GC or use the serial path."
            )

        # Per-doc caches: all layers, k_norm'd + RoPE'd at 0..P_d-1,
        # differentiable to the hypernet (unchanged build_kv_prefix_cache).
        li = self.hyperx_config.layer_indices
        doc_caches = [build_kv_prefix_cache(base, x_e, li) for x_e in x_e_per_doc]

        tot_len = bounds[-1]
        # Real-token RoPE: segment reals at P_d..P_d+len-1 (reproduces the
        # serial cache_position = arange(P_eff, P_eff+len)). Read P_d from x_e
        # (its prefix length), NOT key_cache[0] — under a layer subset layer 0
        # may be unselected (0-length) so key_cache[0] would read 0.
        position_ids = torch.zeros(
            (1, tot_len), dtype=torch.long, device=device
        )
        for s in range(n_seg):
            lo, hi = bounds[s], bounds[s + 1]
            p_d = x_e_per_doc[doc_of_seg[s]].shape[2]
            position_ids[0, lo:hi] = torch.arange(
                p_d, p_d + (hi - lo), device=device
            )

        input_ids = ids_row[:tot_len].unsqueeze(0)
        with kv_prefix_packed_attention(
            base, doc_caches, doc_of_seg, bounds, device
        ):
            out = base(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
            )
        logits = out.logits[0]  # (tot_len, V)
        return [logits[bounds[s]:bounds[s + 1]] for s in range(n_seg)]

    # ---- delegate-to-base properties ----

    @property
    def config(self):
        return self.base_model.config

    @property
    def generation_config(self):
        return self.base_model.generation_config

    @property
    def vocab_size(self):
        # CE path (CrossEntropyTrainer) reads `self.model.vocab_size`;
        # mirror ModulatedPretrainedModel so the same trainer drives
        # both methods (HyperX is not a subclass of it).
        return self.base_model.vocab_size

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    # ---- state_dict (only hypernet is trainable; ctx_encoder + base are frozen) ----

    def state_dict(self, *args, **kwargs):
        if any(p.requires_grad for p in self.ctx_encoder.parameters()):
            raise ValueError("ctx_encoder must be frozen")
        if any(p.requires_grad for p in self.base_model.parameters()):
            raise ValueError("base_model must be frozen")
        sd = self.hyperx.state_dict(*args, **kwargs)
        sd["base_model_name_or_path"] = self.base_model.name_or_path
        sd["hyperx_config"] = self.hyperx_config
        sd["ctx_encoder_args"] = self.ctx_encoder_args
        return sd

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        # Pop metadata
        base_name = state_dict.pop("base_model_name_or_path", None)
        self.hyperx_config = state_dict.pop("hyperx_config", self.hyperx_config)
        self.ctx_encoder_args = state_dict.pop("ctx_encoder_args", self.ctx_encoder_args)
        if base_name is not None and base_name != self.base_model.name_or_path:
            raise ValueError(
                f"base model mismatch: ckpt has {base_name}, "
                f"current is {self.base_model.name_or_path}"
            )
        # Strip torch.compile prefix if present
        clean_sd = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state_dict.items()
        }
        return self.hyperx.load_state_dict(clean_sd, strict=True)

    # ---- alternative constructor ----

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        train: bool = False,
        base_model_kwargs: dict | None = None,
        use_flash_attn: bool = True,
    ) -> "ModulatedPretrainedModelX":
        """Reconstruct from a saved state_dict."""
        hyperx_config: HyperXConfig = state_dict["hyperx_config"]
        ctx_encoder_args: CtxEncoderArguments = state_dict["ctx_encoder_args"]
        base_model_name = state_dict["base_model_name_or_path"]
        base_model = get_model(
            base_model_name,
            train=train,
            requires_grad=False,
            peft_config=None,
            model_kwargs=base_model_kwargs,
            use_flash_attn=use_flash_attn,
        )
        model = cls(base_model, hyperx_config, ctx_encoder_args)
        model.load_state_dict(state_dict)
        if train:
            model.train()
            # Only HyperX is trained. Keep frozen towers deterministic and out
            # of train-mode-only behavior such as dropout/cache warnings.
            model.base_model.eval()
            model.ctx_encoder.eval()
            model.hyperx.train()
            for p in model.hyperx.parameters():
                p.requires_grad_(True)
        else:
            model.eval()
            for p in model.hyperx.parameters():
                p.requires_grad_(False)
        return model


# needed for HF Trainer resume_from_checkpoint under torch 2.6
# (weights_only=True is hardcoded at trainer.py:2860; checkpoints embed
# HyperXConfig via state_dict["hyperx_config"] — see from_state_dict above).
torch.serialization.add_safe_globals([HyperXConfig])
