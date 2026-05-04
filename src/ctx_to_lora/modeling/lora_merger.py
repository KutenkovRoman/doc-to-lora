"""
Utilities for merging / aggregating LoRA adapters coming from multiple chunks.
"""

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Integer
from torch import Tensor


def compute_rank(n_lora, rank):
    return (n_lora + 1) * rank


class ChunkwiseMLPMerger(nn.Module):
    """
    Per-layer shared concat-rank MLP merger.

    For each layer, concatenates chunk LoRA ranks [K, r] -> [K*r],
    then applies a shared linear map [K*r -> r].
    The projection is initialized to exact averaging over chunks.

    Notes:
    - This mode has explicit dependence on K (separate projection per K).
    - Parameters are still shared across layers for a given K.
    """

    def __init__(self, base_rank: int):
        super().__init__()
        self.base_rank = int(base_rank)
        self.proj_by_k = nn.ModuleDict()

    def _build_avg_init_proj(self, n_ctx_chunks: int, device, dtype) -> nn.Linear:
        in_rank = n_ctx_chunks * self.base_rank
        proj = nn.Linear(in_rank, self.base_rank, bias=True, device=device, dtype=dtype)
        with torch.no_grad():
            proj.weight.zero_()
            proj.bias.zero_()
            for t in range(self.base_rank):
                for j in range(n_ctx_chunks):
                    proj.weight[t, j * self.base_rank + t] = 1.0 / float(n_ctx_chunks)
        return proj

    def _get_proj(self, n_ctx_chunks: int, device, dtype) -> nn.Linear:
        key = str(int(n_ctx_chunks))
        if key not in self.proj_by_k:
            self.proj_by_k[key] = self._build_avg_init_proj(n_ctx_chunks, device, dtype)
        proj = self.proj_by_k[key]
        if proj.weight.device != device or proj.weight.dtype != dtype:
            proj = proj.to(device=device, dtype=dtype)
            self.proj_by_k[key] = proj
        return proj

    def forward(self, group_loras: Tensor) -> Tensor:
        if group_loras.shape[0] == 1:
            return group_loras[0]

        K, N_L, r, d = group_loras.shape
        if r != self.base_rank:
            raise RuntimeError(
                f"ChunkwiseMLPMerger rank mismatch: expected r={self.base_rank}, got r={r}"
            )

        concat_rank = (
            group_loras.permute(1, 0, 2, 3)
            .contiguous()
            .reshape(N_L, K * r, d)
        )

        proj = self._get_proj(K, group_loras.device, group_loras.dtype)
        merged = proj(concat_rank.transpose(1, 2)).transpose(1, 2)
        return merged


class ChunkwiseDeepSetsMerger(nn.Module):
    """
    Layer-wise DeepSets merger with shared parameters across layers.

    For each layer l:
      h_{j,l} = phi(x_{j,l}),
      s_l = mean_j h_{j,l},
      ell_{j,l} = rho(h_{j,l} + s_l),
      w_{j,l} = softmax_j(ell_{j,l}),
      L_hat_l = sum_j w_{j,l} L_{j,l}.
    """

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.phi:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        # zero rho -> uniform weights at init
        for layer in self.rho:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, group_loras: Tensor) -> Tensor:
        # group_loras: [K, N_L, r, d]
        if group_loras.shape[0] == 1:
            return group_loras[0]

        K, N_L = group_loras.shape[:2]
        flat = group_loras.reshape(K, N_L, -1)
        stats = torch.stack(
            [
                flat.mean(dim=-1),
                flat.std(dim=-1, unbiased=False),
                flat.abs().mean(dim=-1),
            ],
            dim=-1,
        )  # [K, N_L, 3]

        h = self.phi(stats)  # [K, N_L, H]
        s = h.mean(dim=0, keepdim=True)  # [1, N_L, H]
        logits = self.rho(h + s).squeeze(-1)  # [K, N_L]
        weights = torch.softmax(logits, dim=0)
        return torch.einsum("kn,knrd->nrd", weights, group_loras)


class ChunkwiseAttentionPoolMerger(nn.Module):
    """Layer-wise attention pooling merger with shared parameters across layers."""

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.key = nn.Linear(3, hidden_size)
        self.query = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.key.weight)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.query)

    def forward(self, group_loras: Tensor) -> Tensor:
        # group_loras: [K, N_L, r, d]
        if group_loras.shape[0] == 1:
            return group_loras[0]

        K, N_L = group_loras.shape[:2]
        flat = group_loras.reshape(K, N_L, -1)
        stats = torch.stack(
            [
                flat.mean(dim=-1),
                flat.std(dim=-1, unbiased=False),
                flat.abs().mean(dim=-1),
            ],
            dim=-1,
        )  # [K, N_L, 3]

        keys = self.key(stats)  # [K, N_L, H]
        logits = (keys * self.query.view(1, 1, -1)).sum(dim=-1) / (keys.shape[-1] ** 0.5)
        weights = torch.softmax(logits, dim=0)
        return torch.einsum("kn,knrd->nrd", weights, group_loras)


def _rescale_counts_to_sum(counts: Tensor, target_sum: int) -> Tensor:
    if counts.numel() == 0:
        raise RuntimeError("n_ctx_chunks is empty")
    current_sum = int(counts.sum().item())
    if current_sum <= 0:
        raise RuntimeError(f"Invalid n_ctx_chunks: sum must be > 0, got {current_sum}")
    if current_sum == target_sum:
        return counts

    # Prefer exact integer scale when possible.
    if target_sum % current_sum == 0:
        return counts * (target_sum // current_sum)
    if current_sum % target_sum == 0:
        factor = current_sum // target_sum
        if counts.numel() % factor == 0:
            return counts.reshape(-1, factor).sum(dim=1)

    # Fallback: proportional rescaling with remainder distribution.
    scaled = counts.to(torch.float64) * (float(target_sum) / float(current_sum))
    floored = torch.floor(scaled).to(torch.int64)

    positive_mask = counts > 0
    floored[positive_mask] = torch.clamp_min(floored[positive_mask], 1)

    diff = int(target_sum - floored.sum().item())
    if diff > 0:
        frac = scaled - torch.floor(scaled)
        order = torch.argsort(frac, descending=True)
        for idx in order[:diff]:
            floored[idx] += 1
    elif diff < 0:
        frac = scaled - torch.floor(scaled)
        order = torch.argsort(frac, descending=False)
        remaining = -diff
        for idx in order:
            min_allowed = 1 if counts[idx] > 0 else 0
            can_remove = int(floored[idx].item() - min_allowed)
            if can_remove <= 0:
                continue
            delta = min(can_remove, remaining)
            floored[idx] -= delta
            remaining -= delta
            if remaining == 0:
                break
        if remaining != 0:
            raise RuntimeError(
                "Failed to normalize n_ctx_chunks to target sum: "
                f"target={target_sum}, got={int(floored.sum().item())}"
            )

    if int(floored.sum().item()) != target_sum:
        raise RuntimeError(
            "Failed to normalize n_ctx_chunks to target sum: "
            f"target={target_sum}, got={int(floored.sum().item())}"
        )
    return floored.to(dtype=counts.dtype, device=counts.device)


def combine_lora(
    generated_loras: dict[str, dict[str, Tensor]],
    n_ctx_chunks: Integer[Tensor, "n_ctx"],
    lora_bias: dict[str, dict[str, Tensor]] | None = None,
    scalers: Float[Tensor, "n_ctx"] | None = None,
    bias_scaler: float | None = None,
    merge_mode: str = "stack",
    chunk_mergers: dict[str, dict[str, nn.Module]] | nn.ModuleDict | None = None,
) -> dict[str, dict[str, Tensor]]:
    """Combine per-chunk LoRA weights into one LoRA per context.

    Modes:
    - ``stack`` (default): concatenate ranks (original behavior).
    - ``avg``: element-wise average across chunks.
    - ``mlp``: trainable concat-rank map [K*r -> r], initialized as exact averaging.
    - ``deepsets``: DeepSets-style permutation-invariant weighted merge.
    - ``attnpool``: attention pooling over chunk descriptors.
    """
    if bias_scaler is None:
        bias_scaler = 1
    if merge_mode not in ("stack", "avg", "mlp", "deepsets", "attnpool"):
        raise ValueError(f"Unknown merge_mode: {merge_mode}")

    # Assume all modules share same base rank r
    first_module = next(iter(generated_loras))
    sampled_lora = generated_loras[first_module]["A"]  # [tot_chunks, n_layers, n_modules, r, dim]
    base_rank = sampled_lora.shape[-2]
    device = sampled_lora.device
    dtype = sampled_lora.dtype

    # In multi-device setups, n_ctx_chunks can be local/compressed while
    # generated_loras already contains expanded chunks. Rescale counts to
    # match the actual number of chunk rows without changing proportions.
    tot_chunks = sampled_lora.shape[0]
    declared_chunks = int(n_ctx_chunks.sum().item())
    if declared_chunks <= 0:
        raise RuntimeError(f"Invalid n_ctx_chunks: sum must be > 0, got {declared_chunks}")

    if (scalers is not None) and (n_ctx_chunks.numel() != scalers.numel()):
        # Some distributed paths provide n_ctx_chunks at chunk-level while scalers are context-level
        # Compress/expand n_ctx_chunks to per-context counts
        if n_ctx_chunks.numel() % scalers.numel() == 0:
            factor = n_ctx_chunks.numel() // scalers.numel()
            n_ctx_chunks = n_ctx_chunks.reshape(scalers.numel(), factor).sum(dim=1)
        elif scalers.numel() % n_ctx_chunks.numel() == 0:
            factor = scalers.numel() // n_ctx_chunks.numel()
            n_ctx_chunks = n_ctx_chunks.repeat_interleave(factor)
        else:
            raise RuntimeError(
                f"n_ctx_chunks and scalers are incompatible: "
                f"len(n_ctx_chunks)={n_ctx_chunks.numel()}, len(scalers)={scalers.numel()}"
            )

    # Ensure context chunk counts always match actual generated chunk rows
    n_ctx_chunks = _rescale_counts_to_sum(n_ctx_chunks, tot_chunks)

    if merge_mode in ("avg", "mlp", "deepsets", "attnpool"):
        # rank stays at base_rank (+ one bias slot when a shared bias is present)
        max_rank_needed = base_rank + (base_rank if lora_bias is not None else 0)
    else:
        max_rank_needed = int(compute_rank(n_ctx_chunks.max(), base_rank))

    combined_loras: dict[str, dict[str, Tensor]] = {
        module: {"A": None, "B": None} for module in generated_loras.keys()
    }
    rank_dim = 2
    n_ctx = len(n_ctx_chunks)
    rank_per_group = (n_ctx_chunks * base_rank).tolist()
    bias_tensor = None
    for module_name, module_loras in generated_loras.items():
        for matrix_key in ("A", "B"):
            if lora_bias is not None:
                bias_tensor = lora_bias[module_name][matrix_key]
            loras = module_loras[matrix_key]

            if merge_mode in ("avg", "mlp", "deepsets", "attnpool"):
                # Expand per-context scalers to per-chunk before weighting
                if (scalers is not None) and (matrix_key == "A"):
                    expanded_scalers = torch.repeat_interleave(scalers, n_ctx_chunks)
                    loras = loras * expanded_scalers[:, None, None, None]

                # Split along the tot_chunks dimension, one slice per context
                chunks_per_group = [int(c) for c in n_ctx_chunks.tolist()]
                per_group_loras = loras.split(chunks_per_group, dim=0)

                combined_shape = [n_ctx, loras.shape[1], max_rank_needed, loras.shape[3]]
                combined = torch.zeros(*combined_shape, device=device, dtype=dtype)

                for g, group_loras in enumerate(per_group_loras):
                    # group_loras: [n_ctx_chunks_g, n_layers, r, dim]
                    if merge_mode == "avg":
                        merged = group_loras.mean(dim=0)
                    else:
                        if chunk_mergers is None:
                            raise RuntimeError("trainable merge mode requires chunk_mergers")
                        merged = chunk_mergers[module_name][matrix_key](group_loras)

                    combined[g, :, :base_rank, :] = merged

                    if bias_tensor is not None:
                        combined[g, :, base_rank : base_rank * 2, :] = (
                            bias_tensor * bias_scaler
                        )

            else:
                if (scalers is not None) and (matrix_key == "A"):
                    loras = loras * scalers[:, None, None, None]

                flat_loras = rearrange(
                    loras, "tot_chunks n_layers r dim -> 1 n_layers (tot_chunks r) dim"
                )
                tot_rank = flat_loras.shape[rank_dim]
                expected_rank = int(sum(rank_per_group))
                dynamic_rank_per_group = rank_per_group
                if expected_rank != tot_rank:
                    if expected_rank <= 0 or tot_rank % expected_rank != 0:
                        raise RuntimeError(
                            f"rank_per_group does not match lora rank dimension: "
                            f"sum(rank_per_group)={expected_rank}, total_rank={tot_rank}"
                        )
                    scale = tot_rank // expected_rank
                    dynamic_rank_per_group = [int(r * scale) for r in rank_per_group]

                per_group_deltas = flat_loras.split(dynamic_rank_per_group, dim=rank_dim)

                combined_shape = [n_ctx, *per_group_deltas[0].shape[1:]]
                max_delta_rank = max(dynamic_rank_per_group)
                required_rank = max_delta_rank + (base_rank if bias_tensor is not None else 0)
                combined_shape[rank_dim] = max(max_rank_needed, required_rank)

                combined = torch.zeros(*combined_shape, device=device, dtype=dtype)

                for g, deltas in enumerate(per_group_deltas):
                    combined_rank = deltas.shape[rank_dim]

                    combined[g, :, :combined_rank, :] = deltas

                    if bias_tensor is not None:
                        combined[g, :, combined_rank : combined_rank + base_rank, :] = (
                            bias_tensor * bias_scaler
                        )

            combined_loras[module_name][matrix_key] = combined

    return combined_loras


def combine_coeffs(
    generated_coeffs: Float[Tensor, "tot_chunks n_layers n_modules max_n_ctx_chunks"],
    n_ctx_chunks: Integer[Tensor, "n_ctx"],
) -> Float[Tensor, "n_layers n_modules tot_chunks"]:
    tot_chunks, n_layers, n_modules, _ = generated_coeffs.shape

    combined_coeffs = torch.empty(
        (n_layers, n_modules, tot_chunks),
        device=generated_coeffs.device,
        dtype=generated_coeffs.dtype,
    )

    start = 0
    for n_c in n_ctx_chunks:
        end = start + n_c
        combined_coeffs[:, :, start:end] = generated_coeffs[start:end, :, :, start:end].sum(0)
        start = end

    return combined_coeffs
